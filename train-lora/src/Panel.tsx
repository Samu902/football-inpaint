import React, { useState, ChangeEvent } from 'react';
import { Form, Container, Row, Col, Button, Spinner } from 'react-bootstrap';
import ModelApi from './ModelApi';

interface InputPanelProps {
    modelApi: ModelApi
}

export default function Panel(props: InputPanelProps) {

    const [file, setFile] = useState<File>();
    const [image, setImage] = useState(null);
    const [processing, setProcessing] = useState(false);
    const [error, setError] = useState(false);
    const [completed, setCompleted] = useState(false);

    props.modelApi.onProcessStart.push(onStartProcessing);
    props.modelApi.onProcessEndWithSuccess.push(onEndProcessing);
    props.modelApi.onProcessError.push(onErrorProcessing);

    function onFileChange(e: ChangeEvent<HTMLInputElement>) {
        if (!e.target.files)
            return;

        setFile(e.target.files[0])
        if (e.target.files[0]) {
            const reader = new FileReader();
            reader.onloadend = () => setImage(reader.result);
            reader.readAsDataURL(e.target.files[0]);
            props.modelApi.inputImage = e.target.files[0];
        }
        else {
            setImage(null);
            props.modelApi.inputImage = null;
        }
    }

    function onProcessClick() {
        props.modelApi.startImageProcessing();
    }

    function onStartProcessing() {
        setProcessing(true);
    }

    function onEndProcessing() {
        setProcessing(false);
    }

    function onErrorProcessing() {
        setProcessing(false);
    }

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
        <Container className='my-4 px-5 py-4 border border-3 rounded-2'>
            <h5 className='text-center mt-2 mb-4'>Carica i dati per addestrare il modello di inpainting a disegnare nuove squadre e genera il file LoRa.</h5>
            <Form.Label className='mt-1'>File .zip con immagini di addestramento</Form.Label>
            <Form.Control className='mb-4' type='file' accept=".jpg, .png, .jpeg, .gif, .bmp, .tif, .tiff|image/*" onChange={onFileChange} />
            <Form.Label>Squadra</Form.Label>
            <Form.Control className='mb-4' type='text' onChange={onFileChange} />
            <Form.Label>Iterazioni</Form.Label>
            <Form.Control className='mb-4' type='number' min={0} onChange={onFileChange} />
            <Row className='mt-4'>
                <Col xs={3}>
                    <Button type='button' className='btn btn-primary' disabled={!image || processing} onClick={onProcessClick}>Processa!</Button>
                </Col>
                <Col xs={9}>
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
                    {
                        completed &&
                        <div className='position-absolute top-50 start-50 translate-middle'>
                            <p className='text-danger'>Il tuo modello è pronto.</p>
                            <Button type='button' className='btn btn-success' disabled={!image || processing} onClick={onProcessClick}>Scarica!</Button>
                        </div>
                    }
                </Col>
            </Row>
        </Container>
    );
}