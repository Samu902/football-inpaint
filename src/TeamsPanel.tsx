import React, { useState } from 'react';
import Form from 'react-bootstrap/Form';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';
import CardGroup from 'react-bootstrap/CardGroup';
import TeamCard from './TeamCard';

export default function TeamsPanel() {

    let teams = [1, 2, 3, 4, 5]
    let teamsPerRow = 5

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
                            teams.map((team, key) => {
                                return <TeamCard />
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
                            teams.map((team, key) => {
                                return <TeamCard />
                            })
                        }
                    </CardGroup>
                </Col>
            </Row>
        </Container>
    );
}