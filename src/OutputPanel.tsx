import React, { useState } from 'react';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Button from 'react-bootstrap/Button';
import PlaceholderImage from '/img/placeholder.jpg';
import ModelApi from './ModelApi';

interface OutputPanelProps {
    modelApi: ModelApi
}

export default function OutputPanel(props: OutputPanelProps) {

    const [image, setImage] = useState(null);

    props.modelApi.onProcessStart = onImageStarted;
    props.modelApi.onProcessEnd = onImageCompleted;

    function onImageStarted() {
        setImage(null);
    }

    function onImageCompleted(outputImage: string | ArrayBuffer) {
        setImage(outputImage);
    }

    return (
        <Container className='text-center px-5 py-4 border border-3 rounded-2'>
            <h4 className='mb-4'>Qui apparirà l'immagine processata.</h4>
            <img src={image || PlaceholderImage} width={512} height={256} className='border border-3 rounded-3' />
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