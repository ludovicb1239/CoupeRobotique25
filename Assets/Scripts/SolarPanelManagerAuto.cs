using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using static UnityEngine.Rendering.DebugUI;

public class SolarPanelManagerAuto : MonoBehaviour
{
    // Start is called before the first frame update
    public List<GameObject> solarPanels;
    public GameObject panelAuto;
    List<Vector2> SolarPos;
    private void Start()
    {
        SolarPos = new List<Vector2>();
        foreach(GameObject panel in solarPanels)
        {
            SolarPos.Add(new Vector2(panel.transform.position.x, panel.transform.position.z));
        }
    }
    public void Refresh()
    {
        Vector2 pos = new Vector2(panelAuto.transform.position.x, panelAuto.transform.position.z);
        int n = -1;
        double dist, best = 1;
        for (int i = 0; i < SolarPos.Count; i++)
        {
            dist = Vector2.Distance(pos, SolarPos[i]);
            if (dist < best)
            {
                best = dist;
                n = i;
            }
        }
        if (n != -1)
        {
            solarPanels[n].transform.position = panelAuto.transform.position;
            solarPanels[n].transform.rotation = panelAuto.transform.rotation;
        }
    }
}
