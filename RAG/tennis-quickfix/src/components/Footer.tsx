import React from 'react';

export const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="fixed bottom-0 left-0 right-0 z-50 bg-aws-header h-8 flex items-center justify-center border-t border-aws-border-dark">
      <span className="text-aws-text-primary text-sm">
        © Mercersoft, {currentYear}
      </span>
    </footer>
  );
}; 