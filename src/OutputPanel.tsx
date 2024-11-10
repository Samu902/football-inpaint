import React, { useState } from 'react';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Button from 'react-bootstrap/Button';
import Spinner from 'react-bootstrap/Spinner';
import PlaceholderImage from '/img/placeholder.jpg';
import ModelApi from './ModelApi';

interface OutputPanelProps {
    modelApi: ModelApi
}

export default function OutputPanel(props: OutputPanelProps) {

    const [image, setImage] = useState(null);
    const [processing, setProcessing] = useState(false);
    const [error, setError] = useState(false);

    props.modelApi.onProcessStart.push(onImageStarted);
    props.modelApi.onProcessEndWithSuccess.push(onImageCompleted);
    props.modelApi.onProcessError.push(onImageError);

    function onImageStarted() {
        setImage(null);
        setProcessing(true);
        setError(false);
    }
    
    function onImageCompleted(outputImage: string | ArrayBuffer) {
        setProcessing(false);
        setError(false);
        setImage(outputImage);
    }

    function onImageError(error: string) {
        setImage(null);
        setProcessing(false);
        setError(true);
    }

    return (
        <Container className='text-center px-5 py-4 border border-3 rounded-2'>
            <h4 className='mb-4'>Qui apparirà l'immagine processata.</h4>
            <div className='position-relative'>
                <img src={image || PlaceholderImage} width={512} height={256} className='border border-3 rounded-3' />
                {
                    processing &&
                    <div className='position-absolute top-50 start-50 translate-middle'>
                        <p>Attendi...</p>
                        <Spinner animation="border" role="status" size='sm'>
                            <span className="visually-hidden">Loading...</span>
                        </Spinner>
                    </div>
                }
                {
                    error &&
                    <div className='position-absolute top-50 start-50 translate-middle'>
                        <p className='text-danger'>Errore durante l'elaborazione!</p>
                    </div>
                }
            </div>
            <Row className='mt-4'>
                <Col>
                    {
                        image ? (
                            <a download={props.modelApi.inputImage.name.split('.').slice(0, -1).join('.') + '_processed.png'} href={image}>
                                <Button type='button' className='btn btn-success'>Salva!</Button>
                            </a>
                        ) : (
                            <Button type='button' className='btn btn-success' disabled={true}>Salva!</Button>
                        )
                    }
                </Col>
            </Row>
        </Container>
    );
}