#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
#include <iostream>
using namespace std;


#define ENABLE_VALIDATION

#define WIN_WIDTH 800
#define WIN_HEIGHT 600

#ifdef ENABLE_VALIDATION
const bool enableValidationLayers = true;
#else
const bool enableValidationLayers = false;
#endif

class SimpleApp
{
public:
    void run()
    {
        initVk();
        mainLoop();
        cleanUp();
    }
private:
    void initVk()
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        m_window = glfwCreateWindow(WIN_WIDTH, WIN_HEIGHT, "Vk Simple", nullptr, nullptr);

        createInstance();
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(m_window))
        {
            glfwPollEvents();
        }
    }

    void cleanUp()
    {
        vkDestroyInstance(m_instance, nullptr);

        glfwDestroyWindow(m_window);
        glfwTerminate();
    }

    void createInstance();

    GLFWwindow* m_window = nullptr;
    VkInstance m_instance;
};

void SimpleApp::createInstance()
{
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vk Simple";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "RC Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    auto getRequiredExtensions = []() -> std::vector<const char*>
    {
        uint32_t count = 0;
        const char** extensions = glfwGetRequiredInstanceExtensions(&count);
        std::vector<const char*> requiredExtensions;
        for (uint32_t i = 0; i < count; ++i)
        {
            requiredExtensions.push_back(extensions[i]);
        }
        return requiredExtensions;
    };
    auto requiredExtensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
    createInfo.ppEnabledExtensionNames = requiredExtensions.data();

    const vector<const char*> validationLayers{ "VK_LAYER_KHRONOS_validation" };

    if (enableValidationLayers)
    {
        auto checkValidationLayerSupport = [&]
        {
            uint32_t count = 0;
            vkEnumerateInstanceLayerProperties(&count, nullptr);

            std::vector<VkLayerProperties> availableLayers(count);
            vkEnumerateInstanceLayerProperties(&count, availableLayers.data());
            for (auto layer : validationLayers)
            {
                bool found = false;
                for (auto avail : availableLayers)
                {
                    if (strcmp(avail.layerName, layer) == 0)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    return false;
                }
            }

            return true;
        };

        if (checkValidationLayerSupport())
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
    }

    if (vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create instance!");
    }
}

int main(int argc, char** argv)
{
    SimpleApp app;
    app.run();

    return 0;
}
