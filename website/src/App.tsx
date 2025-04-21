// App.tsx
import React from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import Avatar from './components/Avatar';
import CameraPositionLogger from './components/CameraPositionLogger';
import keypointsData from './assets/test.json';

const App: React.FC = () => {
  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <h1 style={{ position: 'absolute', zIndex: 1, color: '#fff' }}>
        3D Avatar - Tennis Serve
      </h1>
      <Canvas camera={{ position: [0.75, 1.15, 7], fov: 40 }}>
        <color attach="background" args={['#1a1a2e']} />
        <ambientLight intensity={0.8} />
        <directionalLight position={[5, 5, 5]} intensity={1} />
        <Avatar keypointsData={keypointsData} />
        <gridHelper args={[10, 10]} />
        <OrbitControls 
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={1}
          maxDistance={10}
        />
        <CameraPositionLogger />
      </Canvas>
    </div>
  );
};

export default App;
