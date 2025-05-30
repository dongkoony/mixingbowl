package com.mixingbowl.util;

import com.mixingbowl.config.AppConfig;
import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import io.jsonwebtoken.security.Keys;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Date;

@Component
@RequiredArgsConstructor
@Slf4j
public class JwtUtil {

    private final AppConfig appConfig;

    public String generateToken(String email) {
        log.info("genarateToken: {}", email);

        return Jwts.builder()
                .setSubject(email)
                .setIssuedAt(new Date())
                .setExpiration(new Date(System.currentTimeMillis() + appConfig.getJwtExpiration()))
//                .signWith(Keys.hmacShaKeyFor(appConfig.getJwtSecret().getBytes()), SignatureAlgorithm.HS256)
                .signWith(Keys.hmacShaKeyFor(appConfig.getJwtSecret().getBytes()), SignatureAlgorithm.HS256)
                .compact();
    }

    public String getEmailFromToken(String token) {
        try {
            Claims claims = Jwts.parserBuilder()
//                    .setSigningKey(appConfig.getJwtSecret().getBytes())
                    .setSigningKey(Keys.hmacShaKeyFor(appConfig.getJwtSecret().getBytes()))
                    .build()
                    .parseClaimsJws(token)
                    .getBody();
            String email = claims.getSubject();
            log.info("Extracted email from token: {}", email);
            return email;
        } catch (Exception e) {
            log.error("Error extracting email from token: {}", e.getMessage());
            throw new RuntimeException("Invalid token", e);
        }
    }

    public boolean validateToken(String token) {
        try {
            Jwts.parserBuilder()
//                    .setSigningKey(appConfig.getJwtSecret().getBytes())
                    .setSigningKey(Keys.hmacShaKeyFor(appConfig.getJwtSecret().getBytes()))
                    .build()
                    .parseClaimsJws(token);
            return true;
        } catch (Exception e) {
            log.error("Token validation failed: {}", e.getMessage());
            return false;
        }
    }
}
