import React from 'react';
import Container from 'react-bootstrap/Container';
import Panel from './Panel';
import ModelApi from './ModelApi';

export default function App() {

    const modelApi = new ModelApi();

    return (
        <Container>
            <h1 className='text-center my-4'>LoRa training</h1>
            <Panel modelApi={modelApi} />
        </Container>
    );
}
