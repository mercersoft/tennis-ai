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

// COCO keypoint pairs for calculating bone rotations
const BONE_PAIRS = [
  // Torso
  { start: 5, end: 6, bone: ['Spine1', 'spine1'] },  // shoulders
  { start: 11, end: 12, bone: ['Hips', 'hips'] },   // hips
  { start: 5, end: 11, bone: ['LeftUpLeg', 'leftUpLeg'] }, // left torso
  { start: 6, end: 12, bone: ['RightUpLeg', 'rightUpLeg'] }, // right torso

  // Arms
  { start: 5, end: 7, bone: ['LeftArm', 'leftArm'] },     // left upper arm
  { start: 7, end: 9, bone: ['LeftForeArm', 'leftForeArm'] }, // left lower arm
  { start: 6, end: 8, bone: ['RightArm', 'rightArm'] },    // right upper arm
  { start: 8, end: 10, bone: ['RightForeArm', 'rightForeArm'] }, // right lower arm

  // Legs
  { start: 11, end: 13, bone: ['LeftLeg', 'leftLeg'] },    // left upper leg
  { start: 13, end: 15, bone: ['LeftFoot', 'leftFoot'] },  // left lower leg
  { start: 12, end: 14, bone: ['RightLeg', 'rightLeg'] },   // right upper leg
  { start: 14, end: 16, bone: ['RightFoot', 'rightFoot'] }  // right lower leg
];

const Avatar: React.FC<AvatarProps> = ({ keypointsData }) => {
  const { scene } = useGLTF('/avatar.glb');
  const modelRef = useRef<Group>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [bones, setBones] = useState<{ [key: string]: Bone }>({});
  const [hasLoggedKeypoints, setHasLoggedKeypoints] = useState(false);

  // Initialize bones map on model load
  useEffect(() => {
    if (!modelRef.current) return;

    const bonesMap: { [key: string]: Bone } = {};
    modelRef.current.traverse((object) => {
      if (object instanceof Bone) {
        bonesMap[object.name.toLowerCase()] = object;
        console.log('Found bone:', object.name);
      }
    });
    setBones(bonesMap);
  }, [modelRef.current]);

  // Animation loop
  useFrame(() => {
    if (!keypointsData?.frames || !modelRef.current) return;

    // Log first frame keypoints once for debugging
    if (!hasLoggedKeypoints && keypointsData.frames[0]) {
      console.log('First frame keypoints:', keypointsData.frames[0].keypoints);
      console.log('Number of keypoints:', keypointsData.frames[0].keypoints?.length);
      setHasLoggedKeypoints(true);
    }

    // Update current frame
    setCurrentFrame((prev) => (prev + 1) % keypointsData.frames.length);

    const frame = keypointsData.frames[currentFrame];
    if (!frame?.keypoints) {
      console.warn('No keypoints in frame:', currentFrame);
      return;
    }

    // Verify we have enough keypoints for COCO format
    if (frame.keypoints.length < 17) {
      console.warn(`Insufficient keypoints: ${frame.keypoints.length} (expected 17 for COCO format)`);
      return;
    }

    // Update all bone rotations
    BONE_PAIRS.forEach(({ start, end, bone }) => {
      // Safety check for keypoint indices
      if (start >= frame.keypoints.length || end >= frame.keypoints.length) {
        console.warn(`Invalid keypoint indices: ${start}, ${end}. Max index: ${frame.keypoints.length - 1}`);
        return;
      }

      const startPoint = frame.keypoints[start];
      const endPoint = frame.keypoints[end];

      if (!startPoint || !endPoint) {
        console.warn(`Missing keypoints for indices ${start}, ${end}`);
        return;
      }

      if (startPoint.confidence > 0.5 && endPoint.confidence > 0.5) {
        // Find the bone using any of its possible names
        const boneName = bone.find(name => bones[name.toLowerCase()]);
        const targetBone = boneName ? bones[boneName.toLowerCase()] : null;

        if (targetBone) {
          // Calculate rotation based on keypoint positions
          const angle = Math.atan2(
            endPoint.y - startPoint.y,
            endPoint.x - startPoint.x
          );

          // Apply rotation
          targetBone.rotation.z = angle;
        } else {
          console.warn(`Bone not found for names: ${bone.join(', ')}`);
        }
      }
    });
  });

  return <primitive ref={modelRef} object={scene} />;
};

export default Avatar;
