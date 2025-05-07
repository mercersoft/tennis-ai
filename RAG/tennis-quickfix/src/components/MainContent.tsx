import React from 'react';
// import { useGlobalState } from '../contexts/GlobalStateContext'; // Removed this line

interface MainContentProps {
  isDarkMode: boolean;
}

export const MainContent: React.FC<MainContentProps> = ({ isDarkMode }) => {
  // const { activeModel } = useGlobalState(); // Removed this line

  // useEffect(() => { // Removed this block
  // console.log('Active Model:', activeModel); // Debug log
  // }, [activeModel]); // Removed this block

  return (
    <div className="container mx-auto pb-8">
      <div className="flex flex-col items-center justify-center space-y-6">

        {/* {activeModel ? ( // Removed this block
          <div className={`text-center p-4 rounded-lg w-full max-w-md mt-6 ${
            isDarkMode 
              ? 'bg-opacity-10 border border-opacity-20' 
              : 'bg-gray-50 border border-gray-200 shadow-sm'
          }`}>
            <h2 className={`text-2xl font-semibold mb-3 ${isDarkMode ? 'text-aws-text-primary' : 'text-aws-header'}`}>
              Active Model: {activeModel.name}
            </h2>
            <div className={`flex justify-center gap-8 text-lg ${isDarkMode ? 'text-aws-text-secondary' : 'text-gray-600'} 
              ${!isDarkMode && 'border-t border-gray-200 pt-3 mt-3'}`}>
              <div>
                <span className="font-medium">Products:</span> {activeModel.products?.length || 0}
              </div>
              <div>
                <span className="font-medium">Resources:</span> {activeModel.resources?.length || 0}
              </div>
            </div>
          </div>
        ) : ( // Removed this block */} 
          <div className={`text-xl italic ${isDarkMode ? 'text-aws-text-secondary' : 'text-gray-600'}`}>
            {/* No model currently active // Modified this line - or provide a generic message */}
            Main content area.
          </div>
        {/* )} // Removed this line */} 

      </div>
    </div>
  );
}; 