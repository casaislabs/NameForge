"""
Domain Checker Utility
Checks domain availability for generated names.
"""

import requests
import time
import socket
import logging
from typing import List, Dict, Optional
from config import DOMAIN_CHECK, DEFAULT_TLD, TLD_OPTIONS, logger, log_api_call, log_parameters


class DomainChecker:
    """Utility class for checking domain availability."""
    
    def __init__(self):
        """Initialize domain checker with robust configuration."""
        try:
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': DOMAIN_CHECK.get("user_agent", "NameForge/1.0")
            })
            
            # Get configuration with fallbacks
            self.delay = DOMAIN_CHECK.get("delay", 0.1)
            self.timeout = DOMAIN_CHECK.get("timeout", 5.0)
            self.max_concurrent = DOMAIN_CHECK.get("max_concurrent", 5)
            self.retry_attempts = DOMAIN_CHECK.get("retry_attempts", 2)
            
            logger.info(f"DomainChecker initialized with delay={self.delay}s, timeout={self.timeout}s")
            
        except Exception as e:
            logger.error(f"Error initializing DomainChecker: {e}")
            # Set safe defaults
            self.delay = 0.1
            self.timeout = 5.0
            self.max_concurrent = 5
            self.retry_attempts = 2
    
    def check_domain_availability(self, domain_name: str, tld: Optional[str] = None) -> Dict[str, bool]:
        """
        Check if a domain is available for the given TLD.
        """
        try:
            # Log function call
            logger.debug(f"🔍 check_domain_availability called with: {domain_name}")
            
            # Determine TLD to use
            use_tld = (tld or DEFAULT_TLD).lower()
            if not use_tld.startswith("."):
                use_tld = f".{use_tld}"
            if use_tld not in TLD_OPTIONS:
                logger.warning(f"TLD '{use_tld}' not in allowed list {TLD_OPTIONS}; falling back to {DEFAULT_TLD}")
                use_tld = DEFAULT_TLD

            # Log parameters for debugging
            log_parameters("check_domain_availability", {
                "domain_name": domain_name,
                "tld": use_tld,
                "timeout": self.timeout,
                "retry_attempts": self.retry_attempts
            })
            
            if not domain_name or not isinstance(domain_name, str):
                logger.warning(f"Invalid domain name: {domain_name}")
                return {"domain": f"invalid{use_tld}", "available": None}
            
            # Clean domain name
            domain_name = domain_name.strip().lower()
            if not domain_name.isalnum():
                logger.warning(f"Domain name contains invalid characters: {domain_name}")
                return {"domain": f"{domain_name}{use_tld}", "available": None}
            
            full_domain = f"{domain_name}{use_tld}"
            logger.debug(f"Checking domain: {full_domain}")
            
            try:
                # Set socket timeout
                socket.setdefaulttimeout(self.timeout)
                socket.gethostbyname(full_domain)
                # If we get here, domain exists
                logger.debug(f"Domain {full_domain} is taken")
                return {"domain": full_domain, "available": False}
                
            except socket.gaierror:
                # Domain doesn't exist
                logger.debug(f"Domain {full_domain} is available")
                return {"domain": full_domain, "available": True}
                
            except socket.timeout:
                logger.warning(f"Timeout checking domain {full_domain}")
                return {"domain": full_domain, "available": None}
                
        except Exception as e:
            logger.error(f"Error checking domain {domain_name}: {e}")
            
            # Log the error
            log_api_call(
                api_type="domain_check",
                endpoint="check_domain_availability",
                parameters={"domain_name": domain_name, "tld": tld},
                error=e
            )
            
            return {"domain": f"{domain_name}{(tld or DEFAULT_TLD)}", "available": None}
    
    def check_multiple_domains(self, domain_names: List[str], delay: float = None, tld: Optional[str] = None) -> List[Dict[str, bool]]:
        """
        Check multiple domains with a delay to avoid rate limiting.
        """
        try:
            # Log function call
            logger.info(f"🌐 check_multiple_domains called with {len(domain_names)} domains, delay={delay or self.delay}s")
            
            # Determine TLD to use
            use_tld = (tld or DEFAULT_TLD).lower()
            if not use_tld.startswith("."):
                use_tld = f".{use_tld}"
            if use_tld not in TLD_OPTIONS:
                logger.warning(f"TLD '{use_tld}' not in allowed list {TLD_OPTIONS}; falling back to {DEFAULT_TLD}")
                use_tld = DEFAULT_TLD

            # Log parameters for debugging
            log_parameters("check_multiple_domains", {
                "domain_count": len(domain_names),
                "domain_names": domain_names[:5] + ["..."] if len(domain_names) > 5 else domain_names,
                "delay": delay or self.delay,
                "timeout": self.timeout,
                "tld": use_tld
            })
            
            if delay is None:
                delay = self.delay
                
            results = []
            
            for i, domain in enumerate(domain_names):
                logger.debug(f"Checking domain {i+1}/{len(domain_names)}: {domain}")
                result = self.check_domain_availability(domain, tld=use_tld)
                results.append(result)
                
                if delay > 0 and i < len(domain_names) - 1:  # Don't delay after last domain
                    time.sleep(delay)
            
            logger.info(f"✅ Domain checking completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in check_multiple_domains: {e}")
            
            # Log the error
            log_api_call(
                api_type="domain_check",
                endpoint="check_multiple_domains",
                parameters={"domain_names": domain_names, "delay": delay, "tld": tld},
                error=e
            )
            
            return []
    
    def check_domains_across_tlds(self, domain_names: List[str], tlds: Optional[List[str]] = None, delay: float = None) -> Dict[str, List[Dict[str, bool]]]:
        """
        Check availability of the same list of domain names across multiple TLDs.
        
        Args:
            domain_names (List[str]): Names (without TLD) to check
            tlds (Optional[List[str]]): List of TLDs to check (e.g., ['.com', '.ai'])
            delay (float): Delay between checks (uses config default if None)
        
        Returns:
            Dict[str, List[Dict[str, bool]]]: mapping of tld -> list of results
        """
        try:
            # Normalize and validate TLDs
            input_tlds = tlds or [DEFAULT_TLD]
            normalized_tlds: List[str] = []
            for t in input_tlds:
                if not t:
                    continue
                t_str = str(t).strip().lower()
                if not t_str.startswith("."):
                    t_str = f".{t_str}"
                if t_str in TLD_OPTIONS and t_str not in normalized_tlds:
                    normalized_tlds.append(t_str)
            if not normalized_tlds:
                normalized_tlds = [DEFAULT_TLD]
            
            if delay is None:
                delay = self.delay
            
            # Log parameters for debugging
            log_parameters("check_domains_across_tlds", {
                "domain_count": len(domain_names),
                "domain_names": domain_names[:5] + ["..."] if len(domain_names) > 5 else domain_names,
                "tlds": normalized_tlds,
                "delay": delay,
                "timeout": self.timeout
            })
            
            # Perform checks per TLD
            results_by_tld: Dict[str, List[Dict[str, bool]]] = {}
            for tld in normalized_tlds:
                results_by_tld[tld] = self.check_multiple_domains(domain_names, delay=delay, tld=tld)
            
            logger.info(f"✅ Multi-TLD domain checking completed for TLDs: {', '.join(normalized_tlds)}")
            return results_by_tld
        
        except Exception as e:
            logger.error(f"Error in check_domains_across_tlds: {e}")
            log_api_call(
                api_type="domain_check",
                endpoint="check_domains_across_tlds",
                parameters={"domain_names": domain_names, "tlds": tlds, "delay": delay},
                error=e
            )
            return {}
    
    def filter_available_domains(self, domain_names: List[str]) -> List[str]:
        """
        Filter list to only include available domains.
        """
        available = []
        
        for domain in domain_names:
            result = self.check_domain_availability(domain)
            if result["available"]:
                available.append(domain)
        
        return available


def check_domain_availability(domain_name: str, tld: Optional[str] = None) -> bool:
    """
    Simple function to check if a domain is available.
    """
    try:
        logger.debug(f"🔍 check_domain_availability convenience function called with: {domain_name}")
        checker = DomainChecker()
        result = checker.check_domain_availability(domain_name, tld=tld)
        return result["available"]
    except Exception as e:
        logger.error(f"Error in check_domain_availability convenience function: {e}")
        return False


def check_multiple_domains(domain_names: List[str], tld: Optional[str] = None) -> List[Dict[str, bool]]:
    """
    Check multiple domains for availability.
    """
    try:
        logger.debug(f"🌐 check_multiple_domains convenience function called with {len(domain_names)} domains")
        checker = DomainChecker()
        return checker.check_multiple_domains(domain_names, tld=tld)
    except Exception as e:
        logger.error(f"Error in check_multiple_domains convenience function: {e}")
        return []

def check_domains_across_tlds(domain_names: List[str], tlds: Optional[List[str]] = None) -> Dict[str, List[Dict[str, bool]]]:
    """
    Convenience function to check availability of domains across multiple TLDs.
    """
    try:
        checker = DomainChecker()
        return checker.check_domains_across_tlds(domain_names, tlds=tlds)
    except Exception as e:
        logger.error(f"Error in check_domains_across_tlds convenience function: {e}")
        return {}