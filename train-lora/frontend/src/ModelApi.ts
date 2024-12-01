export default class ModelApi {
    
    inputFile: File = null;
    team: string = null;
    steps: number = 0;

    onProcessStart = [() => { return }];
    onProcessSuccess = [(outputFile: string | ArrayBuffer) => { return }];
    onProcessError = [(error: string) => { return }];

    taskID: number = null;
    pollingTime = 15000;

    host: string = 'http://localhost:5002';

    async startLoRaProcessing(): Promise<void> {
        const formData = new FormData();
        formData.append('input_file', this.inputFile);
        formData.append('team', this.team);
        formData.append('steps', this.steps.toString());

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

            const response = await fetch(this.host + '/process-lora/start', {
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
            console.log('Processing LoRa iniziato: taskID=' + this.taskID)
            setTimeout(this.updateLoRaProcessing.bind(this), this.pollingTime);
        } catch (error) {
            this.onProcessError.forEach(f => f(error));
            console.error('Error: ', error);
            throw error;
        }
    }

    async updateLoRaProcessing(): Promise<void> {
        try {
            const response = await fetch(this.host + '/process-lora/update/' + this.taskID, {
                method: 'GET'
            });

            if (!response.ok) {
                const json = await response.json();
                this.onProcessError.forEach(f => f(json.error));
                throw new Error('API Flask error: ' + json.error);
            }
            
            if(response.status == 200) {
                this.finalizeLoRaProcessing()
            }
            else {
                setTimeout(this.updateLoRaProcessing.bind(this), this.pollingTime);
            }
        } catch (error) {
            this.onProcessError.forEach(f => f(error));
            console.error('Error: ', error);
            throw error;
        }
    }

    async finalizeLoRaProcessing(): Promise<void> {
        try {
            const response = await fetch(this.host + '/process-lora/finalize/' + this.taskID, {
                method: 'GET'
            });

            if (!response.ok) {
                const json = await response.json();
                this.onProcessError.forEach(f => f(json.error));
                throw new Error('API Flask error: ' + json.error);

            }

            const blob = await response.blob();
            const reader = new FileReader();
            reader.onloadend = () => this.onProcessSuccess.forEach(f => f(reader.result));
            reader.readAsDataURL(blob);
        } catch (error) {
            this.onProcessError.forEach(f => f(error));
            console.error('Error: ', error);
            throw error;
        }
    }
}
