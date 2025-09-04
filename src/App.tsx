import { useRef, useEffect } from "react";
import {
  createDetector,
  SupportedModels,
  movenet,
} from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl"; // This one's
import "@tensorflow/tfjs-backend-webgpu"; // We just doing this so the code doesn't freak out

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    const setupCamera = async () => {
      if (!videoRef.current) return;
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
    };

    const runPoseDetection = async () => {
      await tf.setBackend("webgl"); // force stable backend
      await tf.ready();
      console.log("TF backend ready:", tf.getBackend());

      const detector = await createDetector(
        SupportedModels.MoveNet,
        {
          modelType: movenet.modelType.SINGLEPOSE_LIGHTNING,
        }  
      );
      console.log("Detector created:", detector);


      const detectPose = async () => {
        console.log("Running pose detection loop...");
        if (!videoRef.current) return;
        const poses = await detector.estimatePoses(videoRef.current);
        if (poses.length > 0) {
          console.log(poses[0].keypoints); // 👀 log keypoints for now
        }
        requestAnimationFrame(detectPose);
      };

      detectPose();
    };

    setupCamera().then(runPoseDetection);
  }, []);

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gray-100">
      <h1 className="text-3xl font-bold text-blue-600 mb-4">
        Fitness Tracker MVP
      </h1>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="border-2 border-black rounded-md"
        width={640}
        height={480}
      />
    </div>
  );
}
