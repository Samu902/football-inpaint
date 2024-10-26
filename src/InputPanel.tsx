import React, { useState } from 'react';
import Form from 'react-bootstrap/Form';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';

export default function InputPanel() {
    
    
    return (
        <Container className='text-center px-5 border border-3 rounded-2'>
            <h4 className='my-4'>Carica l'immagine che vuoi processare.</h4>
            <img src='' width={512} height={256} />
            <Form.Control className='my-4' type='file' />
        </Container>
    );
}