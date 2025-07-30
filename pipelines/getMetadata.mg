{
    "header": {
        "nodesVersions": {
            "Publish": "1.3",
            "CameraInit": "9.0"
        },
        "releaseVersion": "2023.3.0",
        "fileVersion": "1.1",
        "template": true
    },
    "graph": {
        "CameraInit_1": {
            "nodeType": "CameraInit",
            "position": [
                461,
                -61
            ],
            "inputs": {}
        },
        "Publish_1": {
            "nodeType": "Publish",
            "position": [
                647,
                -61
            ],
            "inputs": {
                "inputFiles": [
                    "{CameraInit_1.output}"
                ]
            }
        }
    }
}