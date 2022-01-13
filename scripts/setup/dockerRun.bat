docker run -it -p 7745:7745 --name debugginator-container^
 --env AUTHENTICATE_VIA_JUPYTER="debugginator"^
 -v %cd%:/root/thedebugginator debugginator-image