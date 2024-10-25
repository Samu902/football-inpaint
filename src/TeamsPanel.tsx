import React, { useState } from 'react';
import Form from 'react-bootstrap/Form';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';
import TeamCard from './TeamCard';

export default function TeamsPanel() {

    let teams = [1, 2, 3, 4, 5]
    let teamsPerRow = 5

    return (
        <Container className='py-4'>
            <h5 className='text-center mb-4'>Scegli le due squadre che appariranno nell'immagine finale.</h5>
            <Row>
                {
                    teams.slice(0, 5).map((team, key) => {
                        return <Col><TeamCard /></Col>
                    })
                }
            </Row>
            <Row>
                {
                    teams.slice(5, 10).map((team, key) => {
                        return <Col><TeamCard /></Col>
                    })
                }
            </Row>
        </Container>
    );
}