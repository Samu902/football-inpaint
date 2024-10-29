class ModelApi {

    async processImage(file: File, team1: string, team2: string): Promise<string> {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('team1', team1);
        formData.append('team2', team2);

        try {
            const response = await fetch('http://localhost:5000/process-image', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok)
                throw new Error('Errore durante l\'upload dell\'immagine');

            const blob = await response.blob();
            const outputFileUrl = window.URL.createObjectURL(blob);
            return outputFileUrl;
        } catch (error) {
            console.error('Error:', error);
            throw error;
        }
    }
}