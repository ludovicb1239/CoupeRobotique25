using OpenCVForUnity.ArucoModule;
using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
[System.Serializable]
public struct ArUcoTag
{
    public string name;
    public float size; //in meters
    public bool isFixed;
    public GameObject baseTag;
    public bool3 usingRestrainPos;
    public Vector3 restrainPosition;
    public bool3 usingRestrainRot;
    public Vector3 restrainRotation;
}
[System.Serializable]
public struct listElement
{
    public int ID;
    public ArUcoTag tag;
}
public class ArUcoManager : MonoBehaviour
{
    public static ArUcoManager Instance;
    [SerializeField] private List<listElement> _dict;
    public Dictionary<int, ArUcoTag> dict;
    public float defaultMarkerLength = 0.1f;
    public int TagType = Aruco.DICT_4X4_100;
    public GameObject defaultBaseTag;
    // This method is optional but can be used to initialize the singleton.
    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            dict = new();
            foreach (var item in _dict)
            {
                dict.Add(item.ID, item.tag);
            }
            //DontDestroyOnLoad(gameObject);
        }
    }

}