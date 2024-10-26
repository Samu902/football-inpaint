import React, { useState } from 'react';
import Card from 'react-bootstrap/Card';
import Button from 'react-bootstrap/Button';

import { shirts } from './Shirts';

export default function TeamCard() {

    return (
        <Card style={{ width: '18rem' }}>
            <Card.Img variant="top" src={shirts[0]} width={128} height={128}/>
            <Card.Body>
                <Card.Text className='text-center'>Juventus</Card.Text>
                {/* <Button variant="primary">Go somewhere</Button> */}
            </Card.Body>
        </Card>
    );
}