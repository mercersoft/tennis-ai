import React, { useState, useEffect } from 'react'
import { Header } from './components/Header'
import { Navigation } from './components/Navigation'
import { Footer } from './components/Footer'
import { Chat } from './components/Chat'
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "./components/ui/resizable" 

const AppContent: React.FC = () => {
  const [isDarkMode, setIsDarkMode] = useState(() => {
    // Check if user has a theme preference in localStorage
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      return savedTheme === 'dark';
    }
    // If no saved preference, use system preference
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    // Update document class and localStorage when theme changes
    document.documentElement.classList.toggle('dark', isDarkMode);
    localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

  const handleThemeToggle = () => {
    setIsDarkMode(!isDarkMode);
  };

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden">
      {/* Fixed Header */}
      <Header isDarkMode={isDarkMode} onThemeToggle={handleThemeToggle} />
      
      {/* Main Content Container */}
      <div className="flex flex-1 pt-12 pb-8 overflow-hidden"> {/* Add overflow-hidden here too */}
        <Navigation isDarkMode={isDarkMode} />
        <main className={`
          flex-1
          flex
          flex-col
          overflow-hidden
          transition-all duration-300 ease-in-out
          ${isDarkMode ? 'bg-aws-bg-dark' : 'bg-aws-bg-light'}
          ${isDarkMode ? 'text-aws-header' : 'text-aws-header'}
        `}>
          <ResizablePanelGroup 
            direction="vertical"
            className="min-h-[200px] h-full rounded-lg border"
            autoSaveId="uu_app_main_split" // Use autoSaveId for localStorage persistence
          >
            <ResizablePanel defaultSize={70} minSize={20}>
              {/* Remove items-center and justify-center */}
              <div className="flex h-full"> 
              </div>
            </ResizablePanel>
            <ResizableHandle withHandle />
            <ResizablePanel defaultSize={30} minSize={20}>
               {/* Remove items-center and justify-center */}
              <div className="flex h-full"> 
                <Chat 
                  isDarkMode={isDarkMode} 
                />
              </div>
            </ResizablePanel>
          </ResizablePanelGroup>
        </main>
      </div>

      {/* Fixed Footer */}
      <Footer />
    </div>
  );
}

export default function App() {
  return (
    <AppContent />
  );
}
