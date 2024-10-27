import React, { useState } from 'react';
import './App.css';
import Form from 'react-bootstrap/Form';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';
import InputPanel from './InputPanel';
import OutputPanel from './OutputPanel';
import TeamsPanel from './TeamsPanel';

const shirts = [
    'maglia1.png',
    'maglia2.png',
    'maglia3.png'
];

export default function App() {
    const [image, setImage] = useState(null);
    const [selectedShirt, setSelectedShirt] = useState(null);

    const handleImageUpload = (e: any) => {
        setImage(URL.createObjectURL(e.target.files[0]));
    };

    return (
        <Container>
            <h1 className='text-center my-4'>Football Inpaint</h1>
            <div className='row'>
                <div className='col-6'>
                    <InputPanel />
                </div>
                <div className='col-6'>
                    <OutputPanel />
                </div>
            </div>
            <TeamsPanel />
            {/* <div className="App">
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
        </div> */}
        </Container>
    );
}
