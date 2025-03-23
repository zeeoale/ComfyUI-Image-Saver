import { app } from '../../scripts/app.js'
import { ExifReader } from './lib/exif-reader.js' // https://github.com/mattiasw/ExifReader v4.26.2

app.registerExtension({
    name: "ComfyUI-Image-Saver",
    async setup() {
        // Save original function, reassign to our own handler
        const handleFileOriginal = app.handleFile;
        app.handleFile = async function (file) {
            if (file.type === "image/jpeg" || file.type === "image/webp") {
                try {
                    const exifTags = await ExifReader.load(file);

                    const workflowString = "workflow:";
                    const promptString = "prompt:";
                    let workflow;
                    let prompt;
                    // Search Exif tag data for workflow and prompt
                    Object.values(exifTags).some(value => {
                        try {
                            const description = `${value.description}`;
                            if (workflow === undefined && description.slice(0, workflowString.length).toLowerCase() === workflowString) {
                                workflow = JSON.parse(description.substring(workflowString.length));
                            } else if (prompt === undefined && description.slice(0, promptString.length).toLowerCase() === promptString) {
                                prompt = JSON.parse(description.substring(promptString.length));
                            }
                        } catch (error) {
                            if (!(error instanceof SyntaxError)) {
                                console.error(`ComfyUI-Image-Saver: Error reading Exif value: ${error}`);
                            }
                        }

                        return workflow !== undefined;
                    });

                    if (workflow !== undefined) {
                        // Remove file extension
                        let filename = file.name;
                        let dot = filename.lastIndexOf('.');
                        if (dot !== -1) {
                            filename = filename.slice(0, dot);
                        }

                        app.loadGraphData(workflow, true, true, filename);
                        return;
                    } else if (prompt !== undefined) {
                        app.loadApiJson(prompt);
                        return;
                    }
                } catch (error) {
                    console.error(`ComfyUI-Image-Saver: Error parsing Exif: ${error}`);
                }
            }

            // Fallback to original function
            handleFileOriginal.call(this, file);
        }
    },
})