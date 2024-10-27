import React, { useState } from 'react';
import Card from 'react-bootstrap/Card';

interface TeamCardProps {
    index: number,
    selected: boolean,
    onClickFunction: (i: number) => void,
    shirt: { img: any, team: string }
}

export default function TeamCard(props: TeamCardProps) {

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
        <Card onMouseEnter={e => onMouseEnter(e)} onMouseLeave={e => onMouseLeave(e)} style={{ flex: 'none', cursor: 'pointer' }} onClick={() => props.onClickFunction(props.index)}>
            <img src={props.shirt.img} width={128} height={128} />
            {
                hover || props.selected ? (
                    <>
                        <Card.ImgOverlay className={'p-0 opacity-75 d-flex align-items-center justify-content-center ' + (props.selected ? 'bg-success' : 'bg-secondary')}>
                        </Card.ImgOverlay>
                        <Card.Title className='position-absolute top-50 start-50 translate-middle text-white'>{props.shirt.team}</Card.Title>
                    </>
                ) : ('')
            }
        </Card>
    );
}