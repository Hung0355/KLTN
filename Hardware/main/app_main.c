#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "driver/spi_master.h"
#include "driver/gpio.h"
#include <inttypes.h>
#include "rc522.h"
#include <esp_log.h>
#include "string.h"
#include "math.h"

#include "esp_wifi.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "esp_event.h"
#include "esp_netif.h"

#include "lwip/sockets.h"
#include "lwip/dns.h"
#include "lwip/netdb.h"

#include "esp_log.h"
#include "esp_tls.h"
#include "mqtt_client.h"

static const char* TAG1 = "rc522-demo";
static rc522_handle_t scanner;
static const char *TAG = "MQTT_EXAMPLE";

esp_mqtt_client_handle_t client;

void lcd_init(const unsigned char* msg,int row);
void lcd_cmd(unsigned char);
void lcd_data(unsigned char);
void lcd_decode(unsigned char);
void lcd_string(unsigned char*);

//                  D0,D1,D2,D3,D4,D5,D6,D7,RS,E
int lcd_pins[11] = {15,13,14,16,25,26,27,32,33,17};

void lcd_init(const unsigned char* msg,int row){
    printf("1\n");
    for (int i=0;i<11;i++){
        gpio_pad_select_gpio(lcd_pins[i]);
        gpio_set_direction(lcd_pins[i],GPIO_MODE_OUTPUT);
    }
    
    lcd_cmd(0x38); //configure lcd in 8 bit mode
    lcd_cmd(0x01); // clear display
    lcd_cmd(0x0C); // display on
    if (row == 1)
        lcd_cmd(0x80); // set cursor to 1st line 1position 0xC0(row 2)
    else
        lcd_cmd(0xC0);
    lcd_string(msg); // print string on lcd
}

void lcd_decode(unsigned char info){
    unsigned char temp;
    for(int i = 0;i<8;i++){
        temp=pow(2,i);
        gpio_set_level(lcd_pins[i],(info&temp));
    }
}

void lcd_cmd(unsigned char cmd){
    lcd_decode(cmd);
    gpio_set_level(lcd_pins[8],0);
    gpio_set_level(lcd_pins[9],1);
    vTaskDelay(10/portTICK_PERIOD_MS);
    gpio_set_level(lcd_pins[9],0);
    vTaskDelay(10/portTICK_PERIOD_MS);
}

void lcd_data(unsigned char data){
    lcd_decode(data);
    gpio_set_level(lcd_pins[8],1);
    gpio_set_level(lcd_pins[9],1);
    vTaskDelay(10/portTICK_PERIOD_MS);
    gpio_set_level(lcd_pins[9],0);
    vTaskDelay(10/portTICK_PERIOD_MS);
}

void lcd_string(unsigned char *p){
    while(*p != '\0'){
        if (*p == '/'){
            lcd_cmd(0xC0);
        }
        else{
        lcd_data(*p);}
        p = p + 1;
    }
}

static esp_err_t mqtt_event_handler_cb(esp_mqtt_event_handle_t event)
{
    switch (event->event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "MQTT_EVENT_CONNECTED");
            int msg_id = esp_mqtt_client_subscribe(client, "nodeServer", 0);
            ESP_LOGI(TAG, "sent subscribe successful, msg_id=%d", msg_id);
            unsigned char msg1[25] = "Connect success";
            lcd_init(msg1,1);
            break;
        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "MQTT_EVENT_DISCONNECTED");
            break;
        case MQTT_EVENT_SUBSCRIBED:
            ESP_LOGI(TAG, "MQTT_EVENT_SUBSCRIBED, msg_id=%d", event->msg_id);
            break;
        case MQTT_EVENT_UNSUBSCRIBED:
            ESP_LOGI(TAG, "MQTT_EVENT_UNSUBSCRIBED, msg_id=%d", event->msg_id);
            break;
        case MQTT_EVENT_PUBLISHED:
            ESP_LOGI(TAG, "MQTT_EVENT_PUBLISHED, msg_id=%d", event->msg_id);
            break;
        case MQTT_EVENT_DATA:
            ESP_LOGI(TAG, "MQTT_EVENT_DATA");
            printf("TOPIC=%.*s\r\n", event->topic_len, event->topic);
            printf("DATA=%.*s\r\n", event->data_len, event->data);
            unsigned char msg[32];
            sprintf((char *)msg, "%.*s", event->data_len, event->data);
            lcd_init(msg,1);
            break;
        case MQTT_EVENT_ERROR:
            ESP_LOGI(TAG, "MQTT_EVENT_ERROR");
            break;
        default:
            ESP_LOGI(TAG, "Other event id:%d", event->event_id);
            break;
    }
    return ESP_OK;
}

static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    ESP_LOGD(TAG, "Event dispatched from event loop base=%s, event_id=%d", base, event_id);
    mqtt_event_handler_cb(event_data);
}

static void mqtt_app_start(void)
{
    esp_mqtt_client_config_t mqtt_cfg = {
        .host = "mqtt.flespi.io",
        .port = 1883,
        .username = "FlespiToken yKpDALXhIKxS2HU1iPHd2Xg28UheeZn16T9ohsBTRxTgJO97mlmL7zXUa9gTO8Fa",
        .client_id = "123",
    };

    client = esp_mqtt_client_init(&mqtt_cfg);
    esp_mqtt_client_register_event(client, ESP_EVENT_ANY_ID, mqtt_event_handler, client);
    esp_mqtt_client_start(client);
}

static void wifi_event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGI(TAG, "Disconnected from WiFi, retrying...");
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
    }
}

void wifi_init_sta(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = "hung0355",
            .password = "hungvinh758",
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(ESP_IF_WIFI_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
}

static void rc522_handler(void* arg, esp_event_base_t base, int32_t event_id, void* event_data)
{
    rc522_event_data_t* data = (rc522_event_data_t*) event_data;
    char buffer[21];
    static int i = 1;
    switch(event_id) {
        case RC522_EVENT_TAG_SCANNED: {
                lcd_cmd(0x01);
                rc522_tag_t* tag = (rc522_tag_t*) data->ptr;
                uint64_t value = tag->serial_number;
                sprintf((char *)buffer, "%llu", value);
                esp_mqtt_client_publish(client, "nodeEsp32", buffer, strlen(buffer), 0, 0);
                //sprintf((char *)buffer, "Seat: %d", i++);
                //lcd_init(buffer,2);
                ESP_LOGI(TAG1, "Tag scanned (sn: %" PRIu64 ")", tag->serial_number);
            }
            break;
    }
}

void app_main()
{
    unsigned char msg[25] = "Connecting wifi...";
    lcd_init(msg,1);
    rc522_config_t config = {
        .spi.host = VSPI_HOST,
        .spi.miso_gpio = 19,
        .spi.mosi_gpio = 23,
        .spi.sck_gpio = 18,
        .spi.sda_gpio = 5,
    };
    ESP_ERROR_CHECK(nvs_flash_init());
    wifi_init_sta();
    mqtt_app_start();
    rc522_create(&config, &scanner);
    rc522_register_events(scanner, RC522_EVENT_ANY, rc522_handler, NULL);
    rc522_start(scanner);
}
