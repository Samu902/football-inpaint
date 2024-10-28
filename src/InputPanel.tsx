import React, { useState } from 'react';
import Form from 'react-bootstrap/Form';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Button from 'react-bootstrap/Button';

export default function InputPanel() {
    
    const [filename, setFilename] = useState(null);

    return (
        <Container className='text-center px-5 py-4 border border-3 rounded-2'>
            <h4 className='mb-4'>Carica l'immagine che vuoi processare.</h4>
            <img src='' width={512} height={256} />
            <Row className='mt-4'>
                <Col xs={9}>
                    <Form.Control type='file' value={filename} onChange={e => setFilename(e.currentTarget.value)}/>
                </Col>
                <Col xs={3}>
                    <Button type='button' className='btn btn-primary' onClick={() => alert(filename)}>Processa!</Button>
                </Col>
            </Row>
        </Container>
    );
}