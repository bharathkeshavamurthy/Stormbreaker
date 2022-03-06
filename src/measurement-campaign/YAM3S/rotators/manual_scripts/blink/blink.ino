/*
 As a good start, this script blinks the Arduino onboard LED to verify the board
 is working.

 Yaguang Zhang, Purdue University, 2020/01/31
*/

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  digitalWrite(LED_BUILTIN, HIGH);
  delay(1000);
  digitalWrite(LED_BUILTIN, LOW);
  delay(1000);
}
