import React, { useState } from 'react';
import './App.css';
import Container from 'react-bootstrap/Container';
import InputPanel from './InputPanel';
import OutputPanel from './OutputPanel';
import TeamsPanel from './TeamsPanel';

export default function App() {

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
        </Container>
    );
}
