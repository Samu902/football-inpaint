import React, { useState } from 'react';
import Card from 'react-bootstrap/Card';
import Button from 'react-bootstrap/Button';

import { shirts } from './Shirts';

export default function TeamCard() {

    const [hover, setHover] = useState(false);

    function onMouseEnter(e: any) {
        e.preventDefault();
        setHover(true);
    }

    function onMouseLeave(e: any) {
        e.preventDefault();
        setHover(false);
    }

    return (
        <Card onMouseEnter={e => onMouseEnter(e)} onMouseLeave={e => onMouseLeave(e)} style={{flex: 'none'}} >
            <img src={shirts[0]} width={128} height={128} />
            {
                hover ? (
                    <Card.ImgOverlay className='p-0 bg-secondary opacity-50 d-flex align-items-center justify-content-center'>
                        <Card.Title className='opacity-100'>Juventus</Card.Title>
                    </Card.ImgOverlay>
                ) : ('')
            }
        </Card>
    );
}