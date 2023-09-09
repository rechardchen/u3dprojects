using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

public class BlendTree1DController : MonoBehaviour
{
    Animator animator;
    private int velocityHash;
    private float velocity;
    public float acceleration = 0.05f;

    // Start is called before the first frame update
    void Start()
    {
        animator = GetComponent<Animator>();
        velocityHash = Animator.StringToHash("Velocity");
        velocity = 0;
    }

    // Update is called once per frame
    void Update()
    {
        bool forwardPressed = Input.GetKey("w");

        if (forwardPressed)
        {
            velocity += acceleration * Time.deltaTime;
            if (velocity > 1.0f)
            {
                velocity = 1.0f;
            }
            animator.SetFloat(velocityHash, velocity);
        }

        if (!forwardPressed && velocity > 0.0f)
        {
            velocity  -= acceleration * Time.deltaTime;
            if (velocity < 0.0f)
            {
                velocity = 0.0f;
            }
            animator.SetFloat(velocityHash, velocity);
        }
    }
}
