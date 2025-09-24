/*
  Rui Santos
  Complete project details at https://RandomNerdTutorials.com/esp32-datalogging-google-sheets/
  
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files.
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  Adapted from the examples of the Library Google Sheet Client Library for Arduino devices: https://github.com/mobizt/ESP-Google-Sheet-Client
*/

#include <Arduino.h>
#include <WiFi.h>
#include <Adafruit_Sensor.h>
#include "time.h"
#include <ESP_Google_Sheet_Client.h>

// For SD/SD_MMC mounting helper
#include <GS_SDHelper.h>

//#include "DHT.h"


#include <LiquidCrystal_I2C.h>

#include "DHT.h"

#define DHTPIN 4     // what pin we're connected to

// Uncomment whatever type you're using!
#define DHTTYPE DHT11 



#define WIFI_SSID "unilag"  // INSERT THE NAME OF THE  WIFI YOU ARE USING
#define WIFI_PASSWORD "unilag123"   // INSERT THE PASSWORD OF THE  WIFI YOU ARE USING

// Google Project ID
#define PROJECT_ID "snappy-nomad-457814-t6"

// Service Account's client email
#define CLIENT_EMAIL "google-sheets-datalogging@snappy-nomad-457814-t6.iam.gserviceaccount.com"

#define sensorPower 7
#define sensorPin A0

#define AO_PIN 36

int lcdColumns = 16;
int lcdRows = 2;

// Service Account's private key
const char PRIVATE_KEY[] PROGMEM = "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQC/5qE36dQkn1FJ\nZNNnBCoixLkZtJ/za650xY8RkH5Ud0IJIRKCVBwkaM2W7fe3otv2XwBV7Ynk98RD\n25tZ7AyHVAeNyH6djxIyZ6onhRW1yEKZXxfzDVmY4MID2CYn0lqrchM7o7F/jO5t\n6Xj2adwF9rPZ2FxX2JdgcNPW4NurnSHOnoqkKSzi5C/IY6oe+/AUt0s005YtPbUj\nUpatr6/gx/8+/Q471z/SZVYGwXUBwSk8g12l/sYGaJlwKQNRuu38vUs9AbZr6dpN\nAKpYn9UTx1TH7OsIkoM+HJ0U6wJauDr4+oZAXiIH78Sl/UVjC+7GaYpiXgJo9ayS\niMl64K9BAgMBAAECggEAAijav8zeExgPBZh0qbc4AEx6hOrRvZ8Es24nW642oj2l\n9tFmPnTK7OjYJrSUj/61aw+3Urg7x/Gm/gChRkciDlYyeDC5Vk2GSd59WXDRsWIs\nt/I0TEMmSuJtY4a/ye0f0afs1gwT5OKbmq1828jN4dwsiqko86EBldt6MlP1K+iR\noIBeeZw3kawB9dkUc3Z3htt2T6nFmsKR26DkblkGbVxI3agrBMRbBfHA+mDUKify\nCmDY6elXQkndCXiBsRsDO3ab7XtrhueQNwwYzGYc/9Ey5nPiZZ2BgDY1KMZircFO\nUmRZXSOCZG3OB6hUOQzdbVHrImjPcwtc5hkzoOJdAQKBgQDu+l3DVm9dqICPBvCh\nIcMCVoQ9OMmY8qlNiBF0XtJXiHUueJA5uTW8PAnXtRotTOGy/A65K5H6Gu3/hVPn\nzBLBdIH5FGCX4p8lAH6LksWnTnQspPK38BTqIMJ+zPm09MjMiJ1boWpyWQ2avNfr\nKF8Z0SDor/+GR4VErnH/zeppwQKBgQDNkdaYNpDdbAUl6McHMghIlSAtNoIPLizi\nJvEmN2gtklXtDUQVBKo+I2yPnhKQT45mLvc/WT4QEevtGG3PMjzViWduHMAjd61w\ngFimxiSq/bFoJXtaTsUrcOj2umJz3fk123vZwsIreAQYuHAIPDpygv0AIBhEXMWx\nnBhbl5ilgQKBgDb/4hJx0yGgBS5lr322EnZ4SOj6J9OOjY5tcW7x38ELg8SVNMRE\nLuzKeI1vfb3NQuh5gD33nBoOlpXHLp9bZTmmwb78hJqQKoZOjdE7j0fJE42uzLmu\nhHG55mlebV3LNGd2TZjoFmKIwkvJiHvzh6eebM+AqfNShIZhC5WO45NBAoGAHXW4\n+zdWq2S/mB8z5qQU69HzOoeFPAeyBvbtuDxYga9nAQHVr+1rOFx7Qlm071l3Xs18\nQWfYrRTkaqYFlpvse+2lFPKp1mtgP68lBUGAH8Ebm4FDnD2NpBwaRsGPOmulO0Kp\nDRwMF46rh59m7scy2RZMi6gN0j45Vqq4Eug2yAECgYBmM0GHVoK/kocA0XgqkB2q\nN4/B0xyIfZC57XtBx7lnxHlMz+g1lP+V9kU/LbDLHZmJuNT8pKE0BeqdJeC5GLP4\nNNwSOcBqFCr+y1jNIfuLAVTbV8Fx1LFygy4dbUo/MjqxoXI+VJa8l+1eYjX6DnfJ\ndSI5nMCa0sMr/lYz0to3BQ==\n-----END PRIVATE KEY-----\n";

// The ID of the spreadsheet where you'll publish the data
const char spreadsheetId[] = "1v8bpluiKsy_AOiEkcMyocyZDOD0SBiLKzeVr_R6Dlg8";

// Timer variables
unsigned long lastTime = 0;  
unsigned long timerDelay = 10000;

// Token Callback function
void tokenStatusCallback(TokenInfo info);

// Variables to hold sensor readings
float num = 1200;


// NTP server to request epoch time
const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = 3600;  // 1 hour ahead of UTC for WAT
const int daylightOffset_sec = 0;  // No DST in Nigeria


// Variable to save current epoch time
unsigned long epochTime; 

// Function that gets current epoch time
String getTimeAndDate() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    Serial.println("Failed to obtain time");
    return "N/A";
  }

  // Format the time into a string, like "2025-04-16 15:30:45"
  char timeString[30];
  strftime(timeString, sizeof(timeString), "%Y-%m-%d %H:%M:%S", &timeinfo);
  return String(timeString);
}


DHT dht(4, DHT11);
float Humidity, Temperature;

int _moisture,sensor_analog;
const int sensor_pin = 39;
int lightValue;
float h;
float t;

LiquidCrystal_I2C lcd(0x27, lcdColumns, lcdRows);


void setup(){

    Serial.begin(115200);
    dht.begin();
    Serial.println();

    analogSetAttenuation(ADC_11db);
    digitalWrite(sensorPower, LOW);

    //configTime(0, 0, ntpServer);
    configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);


     lcd.init();
  // turn on LCD backlight                      
     lcd.backlight();
      lcd.setCursor(0, 0);
     lcd.print("  Weather ");
     lcd.setCursor(0, 1);
     lcd.print("  Data Logger");


    GSheet.printf("ESP Google Sheet Client v%s\n\n", ESP_GOOGLE_SHEET_CLIENT_VERSION);

    // Connect to Wi-Fi
    WiFi.setAutoReconnect(true);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
    Serial.print("Connecting to Wi-Fi");
    while (WiFi.status() != WL_CONNECTED) {
      Serial.print(".");
      delay(1000);

     
    }

    Serial.println();
    Serial.print("Connected with IP: ");
    Serial.println(WiFi.localIP());
    Serial.println();

    // Set the callback for Google API access token generation status (for debug only)
    GSheet.setTokenCallback(tokenStatusCallback);

    // Set the seconds to refresh the auth token before expire (60 to 3540, default is 300 seconds)
    GSheet.setPrerefreshSeconds(10 * 60);

    // Begin the access token generation for Google API authentication
    GSheet.begin(CLIENT_EMAIL, PROJECT_ID, PRIVATE_KEY);
    delay(2000);
    
}

void loop(){
    // Call ready() repeatedly in loop for authentication checking and processing
    

    bool ready = GSheet.ready();

    if (ready && millis() - lastTime > timerDelay){
       
       //DisplaySensorsValues();
        DisplaySensorsValues();

        FirebaseJson response;

        Serial.println("\nAppend spreadsheet values...");
        Serial.println("----------------------------");

        FirebaseJson valueRange;

       
        //epochTime = getTime();
        String Tyme = getTimeAndDate();

        valueRange.add("majorDimension", "COLUMNS");
        valueRange.set("values/[0]/[0]", Tyme);
        //valueRange.set("values/[1]/[0]", num);
        valueRange.set("values/[1]/[0]", _moisture);
        valueRange.set("values/[2]/[0]", lightValue);
        valueRange.set("values/[3]/[0]", h);
        valueRange.set("values/[4]/[0]", t);
 //       valueRange.set("values/[2]/[0]", lightValue);
        


        //Serial.print("Light Output: ");
        //Serial.println(lightValue);
      //  Serial.print("Soil Output: ");
	      //Serial.println(soil);

        num = num + 200;
      

        // For Google Sheet API ref doc, go to https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets.values/append
        // Append values to the spreadsheet
        bool success = GSheet.values.append(&response /* returned response */, spreadsheetId /* spreadsheet Id to append */, "Sheet1!A1" /* range to append */, &valueRange /* data range to append */);
        if (success){

            response.toString(Serial, true);
            valueRange.clear();
        }

        else{
            Serial.println(GSheet.errorReason());
        }
        Serial.println();
        Serial.println(ESP.getFreeHeap());
    }

    delay(2000);
    //num += 200;
}

void tokenStatusCallback(TokenInfo info){
    if (info.status == token_status_error){
        GSheet.printf("Token info: type = %s, status = %s\n", GSheet.getTokenType(info).c_str(), GSheet.getTokenStatus(info).c_str());
        GSheet.printf("Token error: %s\n", GSheet.getTokenError(info).c_str());
    }
    else{
        GSheet.printf("Token info: type = %s, status = %s\n", GSheet.getTokenType(info).c_str(), GSheet.getTokenStatus(info).c_str());
    }
}



void DisplaySensorsValues()
{
  
         h = dht.readHumidity();
  // Read temperature as Celsius
         t = dht.readTemperature();

         int h_int = (int)h;
         int t_int = (int)t;

  //
  sensor_analog = analogRead(sensor_pin);
  _moisture = ( 100 - ( (sensor_analog/4095.00) * 100 ) );
 

         lightValue = analogRead(AO_PIN);
        //int soil = readSensor();  
        lastTime = millis();

        Serial.print("Moisture = ");
        Serial.print(_moisture);  /* Print Temperature on the serial window */
        Serial.println("%");
        Serial.print("Light Output: ");
        Serial.println(lightValue);
        Serial.print("Humidity Output: ");
        Serial.println(h);
        Serial.print("Temperature Output: ");
        Serial.println(t);

        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print("Mst: ");
        lcd.setCursor(4, 0);
        lcd.print(_moisture);

         lcd.setCursor(8, 0);
        lcd.print("LI:");
        lcd.setCursor(12, 0);
        lcd.print(lightValue);

        lcd.setCursor(0, 1);
        lcd.print("Hd :");
        lcd.setCursor(4, 1);
        lcd.print(h_int);

        lcd.setCursor(8, 1);
        lcd.print("Tp:");
        lcd.setCursor(12, 1);
        lcd.print(t_int);
        lcd.print("C");
        
         

       /* lcd.setCursor(0, 0);
        lcd.print("Light Intensity: ");
        lcd.setCursor(0, 1);
        lcd.print(lightValue);
        delay(7000);
        lcd.clear();

        lcd.setCursor(0, 0);
        lcd.print("Humidity: ");
        lcd.setCursor(0, 1);
        lcd.print(h_int);
        delay(7000);
        lcd.clear();


        lcd.setCursor(0, 0);
        lcd.print("Temperature: " + t);
        lcd.setCursor(0, 1);
        lcd.print(t_int);
        delay(7000);
        lcd.clear();

        lcd.setCursor(0, 0);
        lcd.print("Humidity: " + h_int);
        lcd.setCursor(0, 1);
        lcd.print("Temperature: " + t_int);
        delay(5000);  */
        
}