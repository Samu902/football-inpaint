export default class ModelApi {
    
    team1: string = 'Juventus';
    team2: string = 'Juventus';
    inputImage: File = null;

    onProcessStart = () => { return };

    onProcessEnd = (outputImage: string | ArrayBuffer) => { return };

    async processImage(): Promise<void> {
        const formData = new FormData();
        formData.append('file', this.inputImage);
        formData.append('team1', this.team1);
        formData.append('team2', this.team2);

        try {
            this.onProcessStart();

            const response = await fetch('http://localhost:5000/process-image', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const json = await response.json();
                throw new Error('API Flask error: ' + json.error);

            }

            const blob = await response.blob();
            const reader = new FileReader();
            reader.onloadend = () => this.onProcessEnd(reader.result);
            reader.readAsDataURL(blob);
        } catch (error) {
            console.error('Error:', error);
            throw error;
        }
    }
}