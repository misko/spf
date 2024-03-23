void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  Serial1.begin(115200);
  Serial1.println("cbw40m_en 0");
  //Serial1.println("wifiscwout 1 12 0"); // channel 12
  Serial1.println("wifiscwout 1 1 30"); // channel 1 , 30*0.25db attenuation
}

void loop() {
   digitalWrite(LED_BUILTIN, LOW);  // turn the LED on (HIGH is the voltage level)
   delay(2000);  
   digitalWrite(LED_BUILTIN, HIGH);  // turn the LED  on (HIGH is the voltage level)
   delay(1000);  
}
