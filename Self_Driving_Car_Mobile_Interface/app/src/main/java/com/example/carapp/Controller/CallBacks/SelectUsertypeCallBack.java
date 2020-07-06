package com.example.carapp.Controller.CallBacks;

import com.example.carapp.Entites.UserType;

public interface SelectUsertypeCallBack {

    void onSuccess(UserType userType);
    void onFailer(String error);

}
