import React, { useState } from 'react';
import './App.css';

const shirts = [
  'maglia1.png',
  'maglia2.png',
  'maglia3.png'
];

function App() {
  const [image, setImage] = useState(null);
  const [selectedShirt, setSelectedShirt] = useState(null);

  const handleImageUpload = (e: any) => {
    setImage(URL.createObjectURL(e.target.files[0]));
  };

  return (
    <div className="App">
      <h1>Carica la tua immagine e seleziona una maglia</h1>
      <input type="file" onChange={handleImageUpload} />
      {image && <img src={image} alt="User Uploaded" className="uploaded-image" />}
      <div className="grid-container">
        {shirts.map((shirt, index) => (
          <img 
            key={index} 
            src={shirt} 
            alt={`Shirt ${index}`} 
            className={`shirt ${selectedShirt === shirt ? 'selected' : ''}`}
            onClick={() => setSelectedShirt(shirt)}
          />
        ))}
      </div>
    </div>
  );
}

export default App;
