import React from 'react';
import './App.css';
import Container from 'react-bootstrap/Container';
import InputPanel from './InputPanel';
import OutputPanel from './OutputPanel';
import TeamsPanel from './TeamsPanel';
import ModelApi from './ModelApi';

export default function App() {

    const modelApi = new ModelApi();

    return (
        <Container>
            <h1 className='text-center my-4'>Football Inpaint</h1>
            <div className='row'>
                <div className='col-6'>
                    <InputPanel modelApi={modelApi}/>
                </div>
                <div className='col-6'>
                    <OutputPanel modelApi={modelApi}/>
                </div>
            </div>
            <TeamsPanel modelApi={modelApi}/>
        </Container>
    );
}
