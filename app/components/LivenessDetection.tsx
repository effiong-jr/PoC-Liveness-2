'use client';

import { useEffect, useRef, useState } from 'react';

const CANVAS_W = 640;
const CANVAS_H = 480;

type AppStatus = 'loading' | 'ready' | 'error';
type FaceResult = { live: boolean; score: number };

export default function LivenessDetection() {
  const videoRef = useRef<HTMLVideoElement>(null);
  // Hidden canvas used as input to FacePlugin SDK
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Visible canvas for video + bounding box overlay
  const overlayRef = useRef<HTMLCanvasElement>(null);

  // SDK session refs — populated once models are loaded
  const sdkRef = useRef<Record<string, unknown> | null>(null);
  const detectSessionRef = useRef<unknown>(null);
  const liveSessionRef = useRef<unknown>(null);

  // Loop control
  const rafRef = useRef<number>(0);
  const runningRef = useRef(false);

  const [appStatus, setAppStatus] = useState<AppStatus>('loading');
  const [loadingMsg, setLoadingMsg] = useState('Initializing...');
  const [isRunning, setIsRunning] = useState(false);
  const [faces, setFaces] = useState<FaceResult[]>([]);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // ── Model initialisation ──────────────────────────────────────────────────
  useEffect(() => {
    let alive = true;

    (async () => {
      try {
        // Must configure ONNX Runtime WASM paths before any InferenceSession is
        // created. We serve the .wasm files from /public (root).
        const ort = await import('onnxruntime-web');
        ort.env.wasm.wasmPaths = '/';
        // Disable multithreading — avoids the need for SharedArrayBuffer /
        // cross-origin isolation headers.
        ort.env.wasm.numThreads = 1;

        setLoadingMsg('Loading OpenCV…');
        const SDK = await import('faceplugin-face-recognition-js');
        sdkRef.current = SDK as unknown as Record<string, unknown>;

        // load_opencv appends a <script> tag; it does not return a promise that
        // resolves when OpenCV is ready, so we poll window.cv ourselves.
        await (SDK as { load_opencv: () => Promise<void> }).load_opencv();
        await waitForOpenCv();

        setLoadingMsg('Loading face detection model…');
        detectSessionRef.current = await (
          SDK as { loadDetectionModel: () => Promise<unknown> }
        ).loadDetectionModel();

        setLoadingMsg('Loading liveness model…');
        liveSessionRef.current = await (
          SDK as { loadLivenessModel: () => Promise<unknown> }
        ).loadLivenessModel();

        if (alive) {
          setAppStatus('ready');
          setLoadingMsg('');
        }
      } catch (err: unknown) {
        console.error('[LivenessDetection] init error', err);
        if (alive) {
          setAppStatus('error');
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

  // ── Detection loop ────────────────────────────────────────────────────────
  async function detect() {
    if (!runningRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const overlay = overlayRef.current;
    const SDK = sdkRef.current as {
      detectFace: (session: unknown, id: string) => Promise<{ size: number; bbox: unknown }>;
      predictLiveness: (
        session: unknown,
        id: string,
        bbox: unknown
      ) => Promise<[number, number, number, number, number][]>;
    } | null;

    if (!video || !canvas || !overlay || !SDK) return;

    const ctx = canvas.getContext('2d');
    const octx = overlay.getContext('2d');
    if (!ctx || !octx) return;

    // Capture the current video frame onto both canvases
    ctx.drawImage(video, 0, 0, CANVAS_W, CANVAS_H);
    octx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    octx.drawImage(video, 0, 0, CANVAS_W, CANVAS_H);

    try {
      const det = await SDK.detectFace(detectSessionRef.current, 'live-canvas');

      if (det.size > 0) {
        const liveResults = await SDK.predictLiveness(
          liveSessionRef.current,
          'live-canvas',
          det.bbox
        );

        const faceResults: FaceResult[] = [];

        for (const [x1, y1, x2, y2, score] of liveResults) {
          const live = score >= 0.3;
          const bw = Math.abs(x2 - x1);
          const bh = Math.abs(y2 - y1);

          faceResults.push({ live, score });

          // Draw bounding box
          octx.strokeStyle = live ? '#22c55e' : '#ef4444';
          octx.lineWidth = 3;
          octx.strokeRect(x1, y1, bw, bh);

          // Draw label
          const label = `${live ? 'LIVE' : 'FAKE'}  ${score.toFixed(2)}`;
          octx.font = 'bold 15px monospace';
          const textY = y1 > 28 ? y1 - 10 : y1 + bh + 20;
          octx.fillStyle = live ? '#16a34a' : '#b91c1c';
          octx.fillRect(x1, textY - 16, octx.measureText(label).width + 8, 20);
          octx.fillStyle = '#fff';
          octx.fillText(label, x1 + 4, textY);
        }

        setFaces(faceResults);
      } else {
        setFaces([]);
      }
    } catch (err) {
      console.warn('[LivenessDetection] detection error', err);
    }

    // Schedule next frame only when still running
    if (runningRef.current) {
      rafRef.current = requestAnimationFrame(detect);
    }
  }

  // ── Camera control ────────────────────────────────────────────────────────
  async function startDetection() {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        // Use ideal values so the browser accepts any resolution the camera
        // supports — exact values can cause AbortError on some drivers.
        video: {
          width: { ideal: CANVAS_W },
          height: { ideal: CANVAS_H },
          facingMode: 'user',
        },
        audio: false,
      });
      const video = videoRef.current!;
      video.srcObject = stream;
      await video.play();

      runningRef.current = true;
      setIsRunning(true);
      rafRef.current = requestAnimationFrame(detect);
    } catch (err) {
      console.error('[LivenessDetection] camera error', err);
      setCameraError(err instanceof Error ? err.message : String(err));
      setIsRunning(false);
    }
  }

  function stopDetection() {
    runningRef.current = false;
    cancelAnimationFrame(rafRef.current);

    const video = videoRef.current;
    if (video?.srcObject) {
      (video.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    }

    // Clear overlay canvas
    const octx = overlayRef.current?.getContext('2d');
    if (octx) octx.clearRect(0, 0, CANVAS_W, CANVAS_H);

    setIsRunning(false);
    setFaces([]);
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-6 bg-zinc-950 p-6">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold tracking-tight text-white">
          Face Liveness Detection
        </h1>
        <p className="mt-1 text-sm text-zinc-500">
          Powered by FacePlugin SDK &mdash; 100% on-device, no data leaves your browser
        </p>
      </div>

      {/* Status banner */}
      {appStatus === 'loading' && (
        <div className="flex items-center gap-3 text-zinc-400">
          <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-zinc-600 border-t-white" />
          <span className="text-sm">{loadingMsg}</span>
        </div>
      )}
      {appStatus === 'error' && (
        <div className="max-w-md rounded-lg border border-red-800 bg-red-950/60 px-4 py-3 text-center text-sm text-red-300">
          Initialization failed: {loadingMsg}
        </div>
      )}

      {/* Video / canvas area */}
      <div className="relative overflow-hidden rounded-2xl border border-zinc-800 shadow-2xl">
        {/* Hidden processing canvas — must have id="live-canvas" for the SDK */}
        <canvas
          id="live-canvas"
          ref={canvasRef}
          width={CANVAS_W}
          height={CANVAS_H}
          className="hidden"
        />

        {/* Visible canvas: shows mirrored video + overlays */}
        <canvas
          ref={overlayRef}
          width={CANVAS_W}
          height={CANVAS_H}
          className="block max-w-full"
          style={{ background: '#09090b' }}
        />

        {/* Hidden video element used as the webcam source */}
        <video ref={videoRef} className="hidden" muted playsInline />

        {/* Placeholder shown before camera starts */}
        {!isRunning && (
          <div className="absolute inset-0 flex items-center justify-center">
            <p className="select-none text-sm text-zinc-700">
              Press &quot;Start Detection&quot; to begin
            </p>
          </div>
        )}
      </div>

      {/* Camera error banner */}
      {cameraError && (
        <div className="flex max-w-md items-start gap-3 rounded-lg border border-amber-700 bg-amber-950/60 px-4 py-3 text-sm text-amber-300">
          <span className="flex-1">
            <strong>Camera unavailable:</strong> {cameraError}
            <br />
            <span className="opacity-70">Make sure no other app is using the camera, then try again.</span>
          </span>
          <button
            onClick={() => setCameraError(null)}
            className="mt-0.5 shrink-0 text-amber-400 hover:text-white"
            aria-label="Dismiss"
          >
            ✕
          </button>
        </div>
      )}

      {/* Per-face results */}
      {faces.length > 0 && (
        <div className="flex flex-wrap justify-center gap-2">
          {faces.map((f, i) => (
            <span
              key={i}
              className={`rounded-full px-3 py-1 text-sm font-semibold ${
                f.live
                  ? 'bg-green-900/50 text-green-300'
                  : 'bg-red-900/50 text-red-300'
              }`}
            >
              Face {i + 1}: {f.live ? 'LIVE' : 'FAKE'} &nbsp;
              <span className="font-mono opacity-70">({f.score.toFixed(3)})</span>
            </span>
          ))}
        </div>
      )}

      {/* Start / Stop button */}
      <button
        onClick={isRunning ? stopDetection : startDetection}
        disabled={appStatus !== 'ready'}
        className={`rounded-full px-8 py-3 text-sm font-semibold transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-offset-zinc-950 ${
          appStatus !== 'ready'
            ? 'cursor-not-allowed bg-zinc-800 text-zinc-600'
            : isRunning
            ? 'bg-red-600 text-white hover:bg-red-500 focus-visible:ring-red-500'
            : 'bg-indigo-600 text-white hover:bg-indigo-500 focus-visible:ring-indigo-500'
        }`}
      >
        {appStatus === 'loading'
          ? 'Loading models…'
          : appStatus === 'error'
          ? 'Initialization failed'
          : isRunning
          ? 'Stop Detection'
          : 'Start Detection'}
      </button>
    </div>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Polls window.cv until OpenCV.js WASM finishes initialising. */
function waitForOpenCv(): Promise<void> {
  return new Promise((resolve) => {
    const id = setInterval(() => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const cvReady = (window as any).cv?.imread;
      if (cvReady) {
        clearInterval(id);
        resolve();
      }
    }, 100);
  });
}
