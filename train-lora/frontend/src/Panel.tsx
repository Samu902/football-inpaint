import React, { useState, ChangeEvent } from 'react';
import { Form, Container, Row, Col, Button, Spinner } from 'react-bootstrap';
import ModelApi from './ModelApi';

interface InputPanelProps {
    modelApi: ModelApi
}

export default function Panel(props: InputPanelProps) {

    const [inputFile, setInputFile] = useState<File>();
    const [team, setTeam] = useState<string>();
    const [steps, setSteps] = useState<number>();

    const [parsedResultFile, setParsedResultFile] = useState(null);

    const [processing, setProcessing] = useState(false);
    const [error, setError] = useState(false);
    const [completed, setCompleted] = useState(false);

    props.modelApi.onProcessStart.push(onStartProcessing);
    props.modelApi.onProcessSuccess.push(onEndProcessing);
    props.modelApi.onProcessError.push(onErrorProcessing);

    function onInputFileChange(e: ChangeEvent<HTMLInputElement>) {
        if (!e.target.files)
            return;

        setInputFile(e.target.files[0])
        if (e.target.files[0]) {
            props.modelApi.inputFile = e.target.files[0];
        }
        else {
            props.modelApi.inputFile = null;
        }
    }

    function onTeamChange(e: ChangeEvent<HTMLInputElement>) {
        setTeam(e.target.value);
        props.modelApi.team = e.target.value;
    }

    function onStepsChange(e: ChangeEvent<HTMLInputElement>) {
        setSteps(parseInt(e.target.value));
        props.modelApi.steps = parseInt(e.target.value);
    }

    function onProcessClick() {
        props.modelApi.startLoRaProcessing();
    }

    function onStartProcessing() {
        setParsedResultFile(null);
        setProcessing(true);
        setError(false);
        setCompleted(false);
    }

    function onEndProcessing(outputFile: string | ArrayBuffer) {
        setProcessing(false);
        setError(false);
        setParsedResultFile(outputFile);
        setCompleted(true);
    }

    function onErrorProcessing(error: string) {
        setParsedResultFile(null);
        setProcessing(false);
        setError(true);
        setCompleted(false);
    }

    return (
        <Container className='my-4 px-5 py-4 border border-3 rounded-2'>
            <h5 className='text-center mt-2 mb-4'>Carica i dati per addestrare il modello di inpainting a disegnare nuove squadre e genera il file LoRa.</h5>
            <Form.Label className='mt-1'>File .zip con immagini di addestramento</Form.Label>
            <Form.Control className='mb-4' type='file' accept=".zip" onChange={onInputFileChange} />
            <Form.Label>Squadra</Form.Label>
            <Form.Control className='mb-4' type='text' onChange={onTeamChange} />
            <Form.Label>Iterazioni</Form.Label>
            <Form.Control className='mb-4' type='number' min={0} onChange={onStepsChange} />
            <div className='mt-4 d-flex'>
                <Button type='button' className='btn btn-primary me-4' disabled={!inputFile || !team || steps == null || steps < 0 || processing} onClick={onProcessClick}>Processa!</Button>
                {
                    processing &&
                    <span className='py-2'>
                        Attendi...&nbsp;&nbsp;
                        <Spinner animation="border" role="status" size='sm'>
                            <span className="visually-hidden">Loading...</span>
                        </Spinner>
                    </span>
                }
                {
                    error &&
                    <span className='py-2'>
                        <span className='text-danger'>Errore durante l'elaborazione!</span>
                    </span>
                }
                {
                    completed &&
                    <>
                        {
                            parsedResultFile? (
                                <span>
                                    Il tuo modello è pronto.&nbsp;&nbsp;&nbsp;&nbsp;
                                    <a href={parsedResultFile}>
                                        <Button type='button' className='btn btn-success'>Scarica!</Button>
                                    </a>
                                </span>
                            ) : (
                                <span className='py-2'>
                                    <span className='text-danger'>Errore durante l'elaborazione!</span>
                                </span>
                            )
                        }
                    </>
                }
            </div>
        </Container>
    );
}