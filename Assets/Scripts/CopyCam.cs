using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CopyCam : MonoBehaviour
{
    public Camera otherCam;
    Camera cam;
    // Start is called before the first frame update
    void Start()
    {
        cam = GetComponent<Camera>();
    }

    // Update is called once per frame
    void Update()
    {
        this.transform.rotation = otherCam.transform.rotation;
        cam.fieldOfView = otherCam.fieldOfView;
    }
}
