import React, { useState, ChangeEvent } from 'react';
import Form from 'react-bootstrap/Form';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Button from 'react-bootstrap/Button';
import PlaceholderImage from '/img/placeholder.jpg';
import ModelApi from './ModelApi';

interface InputPanelProps {
    modelApi: ModelApi
}

export default function InputPanel(props: InputPanelProps) {

    const [file, setFile] = useState<File>();
    const [image, setImage] = useState(null);

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
        alert("processa")
        //props.modelApi.processImage();
    }

    return (
        <Container className='text-center px-5 py-4 border border-3 rounded-2'>
            <h4 className='mb-4'>Carica l'immagine che vuoi processare.</h4>
            <img src={image || PlaceholderImage} width={512} height={256} className='border border-3 rounded-3' />
            <Row className='mt-4'>
                <Col xs={9}>
                    <Form.Control type='file' accept=".jpg, .png, .jpeg, .gif, .bmp, .tif, .tiff|image/*" onChange={onFileChange} />
                </Col>
                <Col xs={3}>
                    <Button type='button' className='btn btn-primary' disabled={!image} onClick={onProcessClick}>Processa!</Button>
                </Col>
            </Row>
        </Container>
    );
}