import React, { useState } from 'react';
import Form from 'react-bootstrap/Form';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';

export default function InputPanel() {
    
    
    return (
        <Container>
            <h4 className='text-center'>Carica l'immagine che vuoi processare.</h4>
            <Form.Control type='file' />
        </Container>
    );
}