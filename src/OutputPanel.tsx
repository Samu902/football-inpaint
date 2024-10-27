import React, { useState } from 'react';
import Form from 'react-bootstrap/Form';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Button from 'react-bootstrap/Button';

export default function OutputPanel() {
    
    return (
        <Container className='text-center px-5 py-4 border border-3 rounded-2'>
            <h4 className='mb-4'>Qui apparirà l'immagine processata.</h4>
            <img src='' width={512} height={256} />
            <Row className='mt-4'>
                <Col xs={9}>
                    <Form.Control type='file' />
                </Col>
                <Col xs={3}>
                    <Button type='button' className='btn btn-success'>Salva!</Button>
                </Col>
            </Row>
        </Container>
    );
}