import React, { useState } from 'react';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';
import CardGroup from 'react-bootstrap/CardGroup';
import TeamCard from './TeamCard';
import { shirts } from './Shirts';
import ModelApi from './ModelApi';

interface TeamsPanelProps {
    modelApi: ModelApi
}

export default function TeamsPanel(props: TeamsPanelProps) {

    const [selectedTeam1, setSelectedTeam1] = useState(0);
    const [selectedTeam2, setSelectedTeam2] = useState(0);

    function updateSelectedTeam1(index: number) {
        setSelectedTeam1(index);
        props.modelApi.team1 = shirts[index].img;
    }

    function updateSelectedTeam2(index: number) {
        setSelectedTeam2(index);
        props.modelApi.team2 = shirts[index].img;
    }

    return (
        <Container className='text-center my-4 px-5 py-4 border border-3 rounded-2'>
            <h5 className='text-center mb-4'>Scegli le due squadre che appariranno nell'immagine finale.</h5>
            <Row className='d-flex align-items-center mb-3'>
                <Col xs={2}>
                    <h6>Squadra 1</h6>
                </Col>
                <Col>
                    <CardGroup className='p-auto'>
                        {
                            shirts.map((team, key) => {
                                return <TeamCard key={key} selected={key == selectedTeam1} shirt={shirts[key]} onClickFunction={() => updateSelectedTeam1(key)} />
                            })
                        }
                    </CardGroup>
                </Col>
            </Row>
            <Row className='d-flex align-items-center'>
                <Col xs={2}>
                    <h6>Squadra 2</h6>
                </Col>
                <Col>
                    <CardGroup className='p-auto'>
                        {
                            shirts.map((team, key) => {
                                return <TeamCard key={key} selected={key == selectedTeam2} shirt={shirts[key]} onClickFunction={() => updateSelectedTeam2(key)} />
                            })
                        }
                    </CardGroup>
                </Col>
            </Row>
        </Container>
    );
}