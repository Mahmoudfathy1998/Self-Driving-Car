package com.example.carapp.Entites;

import java.io.Serializable;

public class UserType  implements Serializable {

    private int id;
    private String userTypeName;

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getUserTypeName() {
        return userTypeName;
    }

    public void setUserTypeName(String userTypeName) {
        this.userTypeName = userTypeName;
    }
}
