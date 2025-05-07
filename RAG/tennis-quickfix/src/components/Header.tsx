import React from 'react';

interface HeaderProps {
  isDarkMode: boolean;
  onThemeToggle: () => void;
}

export const Header: React.FC<HeaderProps> = ({ isDarkMode, onThemeToggle }) => {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-aws-header border-b border-aws-border-dark">
      <div className="container mx-auto px-4 h-12 flex items-center justify-between">
        <div className="flex items-center space-x-8">
          <a href="/" className="flex items-center space-x-2">
            <img 
              src="/tennis-ai.svg" 
              alt="Tennis AI Coach Logo" 
              className="h-6 w-auto"
            />
            <span className="text-aws-text-primary font-semibold">Tennis AI Coach - Quick Fix</span>
          </a>
        </div>
        <div className="flex items-center space-x-6">
          {/* <a href="#" className="text-aws-text-secondary hover:text-aws-text-primary transition-colors">
            Documentation
          </a> */}
          <button
            onClick={onThemeToggle}
            className="text-aws-text-secondary hover:text-aws-text-primary transition-colors"
          >
            {isDarkMode ? 'Light Mode' : 'Dark Mode'}
          </button>
          <button className="text-aws-text-secondary hover:text-aws-text-primary transition-colors">
            Login
          </button>
          <button className="text-aws-text-secondary hover:text-aws-text-primary transition-colors">
            Anonymous User
          </button>
        </div>
      </div>
    </header>
  );
}; 