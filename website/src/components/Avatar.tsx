// Avatar.tsx
import React, { useEffect, useRef, useState } from 'react';
import { useGLTF } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { Bone, Group } from 'three';

interface Keypoint {
  x: number;
  y: number;
  confidence: number;
}

interface Frame {
  keypoints: Keypoint[];
}

interface KeypointsData {
  frames: Frame[];
}

interface AvatarProps {
  keypointsData: KeypointsData;
}

const Avatar: React.FC<AvatarProps> = ({ keypointsData }) => {
  const { scene } = useGLTF('/avatar.glb');
  const modelRef = useRef<Group>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [bones, setBones] = useState<{ [key: string]: Bone }>({});

  // Initialize bones map on model load
  useEffect(() => {
    if (!modelRef.current) return;

    const bonesMap: { [key: string]: Bone } = {};
    modelRef.current.traverse((object) => {
      if (object instanceof Bone) {
        bonesMap[object.name] = object;
        console.log('Found bone:', object.name); // Log each bone name
      }
    });
    setBones(bonesMap);
  }, [modelRef.current]);

  // Animation loop
  useFrame(() => {
    if (!keypointsData?.frames || !modelRef.current) return;

    // Update current frame
    setCurrentFrame((prev) => (prev + 1) % keypointsData.frames.length);

    const frame = keypointsData.frames[currentFrame];
    if (!frame?.keypoints) return;

    // Update bone positions based on keypoints
    // Right arm bones (example mapping - adjust based on your model's bone names)
    const rightShoulder = bones['RightArm'] || bones['rightArm'] || bones['right_arm'];
    const rightElbow = bones['RightForeArm'] || bones['rightForeArm'] || bones['right_forearm'];

    if (rightShoulder && frame.keypoints[5].confidence > 0.5) {
      rightShoulder.rotation.z = Math.atan2(
        frame.keypoints[6].y - frame.keypoints[5].y,
        frame.keypoints[6].x - frame.keypoints[5].x
      );
    }

    if (rightElbow && frame.keypoints[6].confidence > 0.5) {
      rightElbow.rotation.z = Math.atan2(
        frame.keypoints[7].y - frame.keypoints[6].y,
        frame.keypoints[7].x - frame.keypoints[6].x
      ) - rightShoulder?.rotation.z || 0;
    }
  });

  return <primitive ref={modelRef} object={scene} />;
};

export default Avatar;
