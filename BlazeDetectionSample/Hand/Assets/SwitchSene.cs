using UnityEngine;
using UnityEngine.SceneManagement;

public class SwitchSene : MonoBehaviour
{
    public void SwitchScene(int i)
    {
        SceneManager.LoadScene(i);
    }
}
