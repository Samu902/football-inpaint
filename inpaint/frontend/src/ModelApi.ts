export default class ModelApi {
    
    team1: string = 'Juventus';
    team2: string = 'Juventus';
    inputImage: File = null;

    host: string = 'http://localhost:5001';

    onProcessStart = [() => { return }];
    onProcessEndWithSuccess = [(outputImage: string | ArrayBuffer) => { return }];
    onProcessError = [(error: string) => { return }];

    taskID: number = null;
    pollingTime = 15000;

    async startImageProcessing(): Promise<void> {
        const formData = new FormData();
        formData.append('input_image', this.inputImage);
        formData.append('team1', this.team1);
        formData.append('team2', this.team2);

        try {
            this.onProcessStart.forEach(f => f());

            //test connessione a backend
            //await fetch(this.host, {
            //    method: 'GET'
            //});

            //test connessione a backend
            //await fetch(this.host, {
            //    method: 'POST',
            //    body: null
            //});

            // potrei fare un Context al posto di prop
            const response = await fetch(this.host + '/process-image/start', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const json = await response.json();
                this.onProcessError.forEach(f => f(json.error));
                throw new Error('API Flask error: ' + json.error);
            }

            const json = await response.json();
            this.taskID = json.task_id
            console.log('Processing immagine iniziato: taskID=' + this.taskID)
            setTimeout(this.updateImageProcessing.bind(this), this.pollingTime);
        } catch (error) {
            this.onProcessError.forEach(f => f(error));
            console.error('Error: ', error);
            throw error;
        }
    }

    async updateImageProcessing(): Promise<void> {
        try {
            const response = await fetch(this.host + '/process-image/update/' + this.taskID, {
                method: 'GET'
            });

            if (!response.ok) {
                const json = await response.json();
                this.onProcessError.forEach(f => f(json.error));
                throw new Error('API Flask error: ' + json.error);
            }
            
            if(response.status == 200) {
                this.finalizeImageProcessing()
            }
            else {
                setTimeout(this.updateImageProcessing.bind(this), this.pollingTime);
            }
        } catch (error) {
            this.onProcessError.forEach(f => f(error));
            console.error('Error: ', error);
            throw error;
        }
    }

    async finalizeImageProcessing(): Promise<void> {
        try {
            const response = await fetch(this.host + '/process-image/finalize/' + this.taskID, {
                method: 'GET'
            });

            if (!response.ok) {
                const json = await response.json();
                this.onProcessError.forEach(f => f(json.error));
                throw new Error('API Flask error: ' + json.error);

            }

            const blob = await response.blob();
            const reader = new FileReader();
            reader.onloadend = () => this.onProcessEndWithSuccess.forEach(f => f(reader.result));
            reader.readAsDataURL(blob);
        } catch (error) {
            this.onProcessError.forEach(f => f(error));
            console.error('Error: ', error);
            throw error;
        }
    }
}
