'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import Webcam from 'react-webcam';

// ─── Types ────────────────────────────────────────────────────────────────────

type AppStatus = 'loading' | 'ready' | 'running' | 'passed' | 'failed';
type Phase = 'detecting_face' | 'blink' | 'smile';

interface PhaseState {
  phase: Phase;
  /** seconds remaining for the current challenge */
  timeLeft: number;
}

// MediaPipe landmark indices for EAR calculation
// [outerCorner, upperOuter, upperInner, innerCorner, lowerInner, lowerOuter]
const LEFT_EYE = [362, 385, 387, 263, 373, 380];
const RIGHT_EYE = [33, 160, 158, 133, 153, 144];

const EAR_CLOSE_THRESHOLD = 0.20;
const EAR_OPEN_THRESHOLD = 0.25;
const SMILE_THRESHOLD = 0.7;
const FACE_TIMEOUT_S = 10;
const CHALLENGE_TIMEOUT_S = 7;
const FACE_MESH_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh';
const FACE_API_CDN =
  'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@0.22.2/weights';

// ─── Helpers ──────────────────────────────────────────────────────────────────

function dist(
  a: { x: number; y: number; z: number },
  b: { x: number; y: number; z: number }
): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function ear(lm: { x: number; y: number; z: number }[], indices: number[]): number {
  const [p1, p2, p3, p4, p5, p6] = indices.map((i) => lm[i]);
  return (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4));
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function LivenessDetection() {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const runningRef = useRef(false);

  // library refs (loaded dynamically, client-only)
  const faceMeshRef = useRef<unknown>(null);
  const faceapiRef = useRef<unknown>(null);

  // liveness state refs (updated inside animation loop, no re-render needed)
  const blinkStateRef = useRef<'open' | 'closed'>('open');
  const blinkDoneRef = useRef(false);
  const smileDoneRef = useRef(false);
  const phaseRef = useRef<Phase>('detecting_face');
  const phaseStartRef = useRef<number>(Date.now());

  const [appStatus, setAppStatus] = useState<AppStatus>('loading');
  const [loadingMsg, setLoadingMsg] = useState('Initializing…');
  const [phaseState, setPhaseState] = useState<PhaseState>({
    phase: 'detecting_face',
    timeLeft: FACE_TIMEOUT_S,
  });
  const [isRunning, setIsRunning] = useState(false);

  // ── Model initialisation ─────────────────────────────────────────────────

  useEffect(() => {
    let alive = true;

    (async () => {
      try {
        // face-api.js is not SSR-safe; import dynamically
        setLoadingMsg('Loading face expression model…');
        const faceapi = await import('face-api.js');
        await faceapi.nets.tinyFaceDetector.loadFromUri(FACE_API_CDN);
        await faceapi.nets.faceExpressionNet.loadFromUri(FACE_API_CDN);
        faceapiRef.current = faceapi;

        // MediaPipe FaceMesh — also browser-only
        setLoadingMsg('Loading face mesh model…');
        const { FaceMesh } = await import('@mediapipe/face_mesh');
        const fm = new FaceMesh({
          locateFile: (file: string) => `${FACE_MESH_CDN}/${file}`,
        });
        fm.setOptions({
          maxNumFaces: 1,
          refineLandmarks: true,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        // Initialise the model before attaching the results handler
        await fm.initialize();
        faceMeshRef.current = fm;

        if (alive) {
          setAppStatus('ready');
          setLoadingMsg('');
        }
      } catch (err) {
        console.error('[Liveness] init error', err);
        if (alive) {
          setAppStatus('failed');
          setLoadingMsg(err instanceof Error ? err.message : String(err));
        }
      }
    })();

    return () => {
      alive = false;
    };
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      runningRef.current = false;
      cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // ── Detection loop ───────────────────────────────────────────────────────

  const detect = useCallback(async () => {
    if (!runningRef.current) return;

    const video = webcamRef.current?.video;
    const canvas = canvasRef.current;
    const faceapi = faceapiRef.current as typeof import('face-api.js') | null;
    const faceMesh = faceMeshRef.current as {
      send: (input: { image: HTMLVideoElement }) => Promise<void>;
      onResults: (cb: (r: FaceMeshResults) => void) => void;
    } | null;

    if (!video || !canvas || !faceapi || !faceMesh) {
      rafRef.current = requestAnimationFrame(detect);
      return;
    }

    if (video.readyState < 2) {
      rafRef.current = requestAnimationFrame(detect);
      return;
    }

    const elapsed = (Date.now() - phaseStartRef.current) / 1000;
    const phase = phaseRef.current;
    const timeout = phase === 'detecting_face' ? FACE_TIMEOUT_S : CHALLENGE_TIMEOUT_S;
    const timeLeft = Math.max(0, Math.ceil(timeout - elapsed));

    setPhaseState({ phase, timeLeft });

    // Timeout → fail
    if (elapsed > timeout) {
      runningRef.current = false;
      setIsRunning(false);
      setAppStatus('failed');
      return;
    }

    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // ── MediaPipe: blink detection ──────────────────────────────────────

    let blinkDetected = false;
    let landmarkCount = 0;

    await new Promise<void>((resolve) => {
      (faceMesh as { onResults: (cb: (r: FaceMeshResults) => void) => void }).onResults(
        (results: FaceMeshResults) => {
          landmarkCount = results.multiFaceLandmarks?.length ?? 0;
          if (landmarkCount > 0 && (phase === 'blink' || phase === 'detecting_face')) {
            const lm = results.multiFaceLandmarks[0];
            const avgEar = (ear(lm, LEFT_EYE) + ear(lm, RIGHT_EYE)) / 2;

            // Draw EAR debug on canvas
            ctx.fillStyle = 'rgba(0,0,0,0.5)';
            ctx.fillRect(4, 4, 160, 24);
            ctx.fillStyle = '#fff';
            ctx.font = '12px monospace';
            ctx.fillText(`EAR: ${avgEar.toFixed(3)}`, 10, 20);

            if (phase === 'blink') {
              if (avgEar < EAR_CLOSE_THRESHOLD) {
                blinkStateRef.current = 'closed';
              } else if (
                avgEar > EAR_OPEN_THRESHOLD &&
                blinkStateRef.current === 'closed'
              ) {
                blinkStateRef.current = 'open';
                blinkDetected = true;
              }
            }
          }
          resolve();
        }
      );
      faceMesh.send({ image: video }).catch(() => resolve());
    });

    // ── face-api.js: smile + face presence ──────────────────────────────

    let facePresent = landmarkCount > 0;
    let smileDetected = false;

    if (phase === 'smile' || phase === 'detecting_face') {
      const detection = await faceapi
        .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
        .withFaceExpressions();

      if (detection) {
        facePresent = true;
        const happy = detection.expressions.happy;

        if (phase === 'smile' && happy > SMILE_THRESHOLD) {
          smileDetected = true;
        }

        // Draw expression bar
        ctx.fillStyle = 'rgba(0,0,0,0.5)';
        ctx.fillRect(4, 32, 160, 24);
        ctx.fillStyle = happy > SMILE_THRESHOLD ? '#22c55e' : '#fff';
        ctx.font = '12px monospace';
        ctx.fillText(`Smile: ${(happy * 100).toFixed(0)}%`, 10, 48);
      }
    }

    // ── Phase transitions ─────────────────────────────────────────────────

    if (phase === 'detecting_face' && facePresent) {
      phaseRef.current = 'blink';
      phaseStartRef.current = Date.now();
    } else if (phase === 'blink' && (blinkDetected || blinkDoneRef.current)) {
      blinkDoneRef.current = true;
      phaseRef.current = 'smile';
      phaseStartRef.current = Date.now();
    } else if (phase === 'smile' && (smileDetected || smileDoneRef.current)) {
      smileDoneRef.current = true;
      runningRef.current = false;
      setIsRunning(false);
      setAppStatus('passed');
      return;
    }

    rafRef.current = requestAnimationFrame(detect);
  }, []);

  // ── Camera control ───────────────────────────────────────────────────────

  function startDetection() {
    blinkStateRef.current = 'open';
    blinkDoneRef.current = false;
    smileDoneRef.current = false;
    phaseRef.current = 'detecting_face';
    phaseStartRef.current = Date.now();
    runningRef.current = true;
    setAppStatus('running');
    setIsRunning(true);
    rafRef.current = requestAnimationFrame(detect);
  }

  function reset() {
    runningRef.current = false;
    cancelAnimationFrame(rafRef.current);
    setIsRunning(false);
    setAppStatus('ready');
    const ctx = canvasRef.current?.getContext('2d');
    ctx?.clearRect(0, 0, canvasRef.current!.width, canvasRef.current!.height);
  }

  // ── Render ───────────────────────────────────────────────────────────────

  const phaseLabel: Record<Phase, string> = {
    detecting_face: 'Looking for your face…',
    blink: 'Please BLINK',
    smile: 'Please SMILE',
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-6 bg-zinc-950 p-6">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold tracking-tight text-white">
          Face Liveness Detection
        </h1>
        <p className="mt-1 text-sm text-zinc-500">
          MediaPipe · face-api.js · react-webcam &mdash; 100% on-device
        </p>
      </div>

      {/* Loading / init error */}
      {appStatus === 'loading' && (
        <div className="flex items-center gap-3 text-zinc-400">
          <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-zinc-600 border-t-white" />
          <span className="text-sm">{loadingMsg}</span>
        </div>
      )}

      {/* Camera + canvas overlay */}
      <div className="relative overflow-hidden rounded-2xl border border-zinc-800 shadow-2xl">
        <Webcam
          ref={webcamRef}
          audio={false}
          mirrored
          videoConstraints={{ width: 640, height: 480, facingMode: 'user' }}
          className="block"
          style={{ width: 640, height: 480 }}
        />

        {/* Debug overlay canvas */}
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          className="pointer-events-none absolute inset-0"
        />

        {/* Phase instruction overlay */}
        {appStatus === 'running' && (
          <div className="absolute bottom-0 left-0 right-0 bg-black/60 px-4 py-3 text-center">
            <p className="text-lg font-semibold text-white">
              {phaseLabel[phaseState.phase]}
            </p>
            <p className="mt-0.5 text-sm text-zinc-400">
              {phaseState.timeLeft}s remaining
            </p>
          </div>
        )}

        {/* Idle overlay before start */}
        {(appStatus === 'ready' || appStatus === 'loading') && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50">
            <p className="select-none text-sm text-zinc-400">
              {appStatus === 'loading'
                ? 'Loading models…'
                : 'Press Start to begin'}
            </p>
          </div>
        )}
      </div>

      {/* Result banners */}
      {appStatus === 'passed' && (
        <div className="flex items-center gap-3 rounded-xl border border-green-700 bg-green-950/60 px-6 py-4 text-green-300">
          <span className="text-2xl">✓</span>
          <div>
            <p className="font-semibold">Liveness Confirmed</p>
            <p className="text-sm opacity-70">Real person detected</p>
          </div>
        </div>
      )}

      {appStatus === 'failed' && (
        <div className="flex items-center gap-3 rounded-xl border border-red-700 bg-red-950/60 px-6 py-4 text-red-300">
          <span className="text-2xl">✗</span>
          <div>
            <p className="font-semibold">Liveness Check Failed</p>
            <p className="text-sm opacity-70">
              Challenge not completed in time
            </p>
          </div>
        </div>
      )}

      {/* Action button */}
      {(appStatus === 'ready' || appStatus === 'passed' || appStatus === 'failed') && (
        <button
          onClick={
            appStatus === 'ready' ? startDetection : reset
          }
          className={`rounded-full px-8 py-3 text-sm font-semibold transition-all ${
            appStatus === 'ready'
              ? 'bg-indigo-600 text-white hover:bg-indigo-500'
              : appStatus === 'passed'
              ? 'bg-green-700 text-white hover:bg-green-600'
              : 'bg-zinc-700 text-white hover:bg-zinc-600'
          }`}
        >
          {appStatus === 'ready' ? 'Start Detection' : 'Try Again'}
        </button>
      )}

      {appStatus === 'running' && (
        <button
          onClick={reset}
          className="rounded-full bg-zinc-800 px-6 py-2 text-sm text-zinc-400 hover:bg-zinc-700"
        >
          Cancel
        </button>
      )}
    </div>
  );
}

// ─── Local type for MediaPipe results ─────────────────────────────────────────

interface FaceMeshResults {
  multiFaceLandmarks: { x: number; y: number; z: number }[][];
}
