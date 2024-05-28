using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class randomiser : MonoBehaviour
{
    [SerializeField] public Vector3 posMin;
    [SerializeField] public Vector3 posMax;
    [SerializeField] public Vector3 rotMin;
    [SerializeField] public Vector3 rotMax;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        this.transform.position = random(posMin, posMax);
        transform.rotation = Quaternion.Euler(random(rotMin, rotMax));
    }
    Vector3 random(Vector3 min, Vector3 max)
    {
        Vector3 ret = new();
        ret.x = Random.Range(min.x, max.x);
        ret.y = Random.Range(min.y, max.y);
        ret.z = Random.Range(min.z, max.z);
        return ret;
    }
}
