// Avatar.tsx
import React from 'react';
import { useGLTF } from '@react-three/drei';

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

const Avatar: React.FC<AvatarProps> = () => {
  // Load the GLTF model
  const { scene } = useGLTF('/avatar.glb');
  
  return (
    <primitive object={scene} />
  );
};

export default Avatar;
