import React, { useState } from 'react';
import { useThree } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import { PerspectiveCamera } from 'three';

const CameraPositionLogger: React.FC = () => {
  const { camera } = useThree();
  const [pos, setPos] = useState([0, 0, 0]);
  const [fov, setFov] = useState(40);

  React.useEffect(() => {
    const updatePos = () => {
      setPos([
        Number(camera.position.x.toFixed(2)),
        Number(camera.position.y.toFixed(2)),
        Number(camera.position.z.toFixed(2))
      ]);
      if (camera instanceof PerspectiveCamera) {
        setFov(Number(camera.fov.toFixed(2)));
      }
    };
    
    // Update initially and on every frame
    updatePos();
    window.requestAnimationFrame(function animate() {
      updatePos();
      window.requestAnimationFrame(animate);
    });
  }, [camera]);

  return (
    <Html
      style={{
        bottom: '-80px',
        left: '-40vw',
        color: 'white',
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        padding: '10px',
        borderRadius: '5px',
        fontFamily: 'monospace',
        whiteSpace: 'nowrap'
      }}
    >
      Camera Position: [{pos.join(', ')}]<br/>
      FOV: {fov}
    </Html>
  );
};

export default CameraPositionLogger; 