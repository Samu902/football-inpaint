import React, { useState } from 'react';
import Card from 'react-bootstrap/Card';
import Button from 'react-bootstrap/Button';

export default function TeamCard() {

    return (
        <Card style={{ width: '18rem' }}>
            <Card.Img variant="top" src="img/juve_shirt.jpg" />
            <Card.Body>
                <Card.Text className='text-center'>Juventus</Card.Text>
                {/* <Button variant="primary">Go somewhere</Button> */}
            </Card.Body>
        </Card>
    );
}